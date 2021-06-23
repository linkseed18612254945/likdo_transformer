from transformers import BertPreTrainedModel, BertModel, RobertaForSequenceClassification, BartModel, BartPretrainedModel, BartConfig
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


def zero_shot_data_collator(features):
    first = features[0]
    batch = {}
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k == "label_positions":
            batch_labels_position = [[] for _ in range(len(first["label_positions"]))]
            for feature in features:
                for i, p in enumerate(feature['label_positions']):
                    batch_labels_position[i].append(p)
            batch_labels_position = [torch.tensor(b, dtype=torch.long) for b in batch_labels_position]
            batch[k] = batch_labels_position
        if k not in ("label", "label_positions") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    return batch


@dataclass
class ZeroShotOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    anchor_vector: Optional[torch.FloatTensor] = None
    label_vectors: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class BertMetricLearningModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.metric_hidden_size = 256
        self.scl_t = 1

        self.ce_p = 0.8
        self.scl_p = 0.1
        self.lscl_p = 0.1

        assert (self.ce_p + self.scl_p + self.lscl_p) == 1

        # self.metric_linear = nn.Sequential(
        #     nn.Linear(config.hidden_size, self.metric_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.LayerNorm(self.metric_hidden_size),
        #     nn.Linear(self.metric_hidden_size, self.metric_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.LayerNorm(self.metric_hidden_size),
        # )

        self.metric_linear = nn.Linear(config.hidden_size, self.metric_hidden_size)
        # self.label_metric_linear = nn.Linear(config.hidden_size, self.metric_hidden_size)
        # self.predict_linear = nn.Linear(self.metric_hidden_size * 2, )

        self.ce_loss_fct = CrossEntropyLoss()
        self.init_weights()

    def label_distance_loss_fct(self, label_vectors):
        distance_loss = 0
        for i in range(label_vectors.shape[1] - 1):
            a = label_vectors[:, i, :].unsqueeze(dim=1)
            b = label_vectors[:, range(i + 1, label_vectors.shape[1]), :]
            similarity = torch.cosine_similarity(a, b, dim=2)
            distance_loss += torch.mean(similarity)
        distance_loss /= label_vectors.shape[1]
        return distance_loss

    def center_loss_fct(self, anchor_vector, labels):
        cl = 0
        for l in set(labels):
            label_vectors = anchor_vector[labels == l, :]
            centor_vector = torch.mean(label_vectors, axis=0)
            l_loss = torch.sum((label_vectors - centor_vector) ** 2) / 2
            cl += l_loss
        return cl / len(set(labels))

    def scl_func(self, anchor_vectors, labels):
        """
        <<SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINED LANGUAGE MODEL FINE-TUNING>>
        :param anchor_vector: batch_size * hidden_size
        :param labels:
        :return:
        """

        total_losses = 0
        anchor_vectors = anchor_vectors.squeeze(dim=1)
        for i in range(anchor_vectors.shape[0]):
            anchor_vector = anchor_vectors[i, :]
            # other_index = torch.from_numpy(np.tile(np.array(list(filter(lambda x: x != i, range(anchor_vectors.shape[0])))),
            #                                        anchor_vectors.shape[1]).reshape(anchor_vectors.shape[1], -1))
            # other_vectors = torch.gather(anchor_vectors.transpose(1, 0), dim=1, index=other_index).transpose(1, 0)

            other_vectors = np.delete(anchor_vectors.detach().cpu(), i, 0).to(anchor_vector.device)
            same_labels = torch.where(labels == labels[i])
            same_label_vectors = anchor_vectors[same_labels]
            if same_label_vectors.shape[0] > 0:
                up = torch.exp(torch.cosine_similarity(same_label_vectors, anchor_vector.unsqueeze(0)) / self.scl_t)
                down = torch.sum(torch.exp(torch.cosine_similarity(other_vectors, anchor_vector.unsqueeze(0)) / self.scl_t))
                singe_sample_loss = torch.sum(torch.log(up/down)) / -(anchor_vectors.shape[0] - 1)
                total_losses += singe_sample_loss
        return total_losses

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            label_positions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        label_max_position = torch.max(label_positions[-1]).tolist()

        token_type_ids = torch.cat((torch.ones((input_ids.shape[0], label_max_position), dtype=torch.long, device=token_type_ids.device),
                                    torch.zeros((input_ids.shape[0], input_ids.shape[1] - label_max_position), dtype=torch.long, device=token_type_ids.device)), dim=1)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        attentions = None
        anchor_vector = sequence_output[:, 0, :].unsqueeze(dim=1)

        # anchor_vector = outputs[1]

        label_vectors = None
        for positions in label_positions:
            position = positions[0]
            label_vector = sequence_output[:, position, :]
            label_vector = torch.mean(label_vector, dim=1).unsqueeze(dim=1)
            if label_vectors is None:
                label_vectors = label_vector
            else:
                label_vectors = torch.cat([label_vectors, label_vector], dim=1)

        label_vectors = self.dropout(label_vectors)
        # anchor_vector = torch.cat([])

        anchor_vector = self.metric_linear(anchor_vector)
        label_vectors = self.metric_linear(label_vectors)
        logits = torch.cosine_similarity(label_vectors, anchor_vector, dim=2)

        loss = None
        if labels is not None:
            ce_loss = self.ce_loss_fct(logits, labels)
            scl_loss = self.scl_func(anchor_vector.squeeze(dim=1), labels) / 10
            # true_label_vectors = label_vectors[range(len(labels)), labels, :]
            # scl_label_loss = self.scl_func(true_label_vectors, labels) / 10

            # center_loss = self.center_loss_fct(anchor_vector, labels)
            # label_distance_loss = self.label_distance_loss_fct(label_vectors)

            loss = ce_loss * self.ce_p + scl_loss * self.scl_p
            # loss = ce_loss * self.ce_p + scl_loss * self.scl_p + scl_label_loss * self.lscl_p
            # loss = ce_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return ZeroShotOutput(
            loss=loss, logits=logits, anchor_vector=anchor_vector, label_vectors=label_vectors, hidden_states=outputs.hidden_states, attentions=attentions,
        )


class BartMetricLearningModel(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.metric_hidden_size = 256
        self.metric_linear = nn.Linear(config.hidden_size, self.metric_hidden_size)
        # self.label_metric_linear = nn.Linear(config.hidden_size, self.metric_hidden_size)
        # self.predict_linear = nn.Linear(self.metric_hidden_size * 2, )
        self.scl_t = 1

        self.ce_p = 0.8
        self.scl_p = 0.1
        self.lscl_p = 0.1
        self.ce_loss_fct = CrossEntropyLoss()

        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)
    def scl_func(self, anchor_vectors, labels):
        """
        <<SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINED LANGUAGE MODEL FINE-TUNING>>
        :param anchor_vector: batch_size * hidden_size
        :param labels:
        :return:
        """

        total_losses = 0
        anchor_vectors = anchor_vectors.squeeze(dim=1)
        for i in range(anchor_vectors.shape[0]):
            anchor_vector = anchor_vectors[i, :]
            # other_index = torch.from_numpy(np.tile(np.array(list(filter(lambda x: x != i, range(anchor_vectors.shape[0])))),
            #                                        anchor_vectors.shape[1]).reshape(anchor_vectors.shape[1], -1))
            # other_vectors = torch.gather(anchor_vectors.transpose(1, 0), dim=1, index=other_index).transpose(1, 0)

            other_vectors = np.delete(anchor_vectors.detach().cpu(), i, 0).to(anchor_vector.device)
            same_labels = torch.where(labels == labels[i])
            same_label_vectors = anchor_vectors[same_labels]
            if same_label_vectors.shape[0] > 0:
                up = torch.exp(torch.cosine_similarity(same_label_vectors, anchor_vector.unsqueeze(0)) / self.scl_t)
                down = torch.sum(torch.exp(torch.cosine_similarity(other_vectors, anchor_vector.unsqueeze(0)) / self.scl_t))
                singe_sample_loss = torch.sum(torch.log(up/down)) / -(anchor_vectors.shape[0] - 1)
                total_losses += singe_sample_loss
        return total_losses

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        label_positions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        label_max_position = torch.max(label_positions[-1]).tolist()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = sequence_output[eos_mask, :].view(sequence_output.size(0), -1, sequence_output.size(-1))[
            :, -1, :
        ]

        anchor_vector = sentence_representation.unsqueeze(dim=1)

        label_vectors = None
        for positions in label_positions:
            position = positions[0]
            label_vector = sequence_output[:, position, :]
            label_vector = torch.mean(label_vector, dim=1).unsqueeze(dim=1)
            if label_vectors is None:
                label_vectors = label_vector
            else:
                label_vectors = torch.cat([label_vectors, label_vector], dim=1)

        anchor_vector = self.metric_linear(anchor_vector)
        label_vectors = self.metric_linear(label_vectors)
        logits = torch.cosine_similarity(label_vectors, anchor_vector, dim=2)

        loss = None
        if labels is not None:
            ce_loss = self.ce_loss_fct(logits, labels)
            scl_loss = self.scl_func(anchor_vector.squeeze(dim=1), labels) / 10
            # true_label_vectors = label_vectors[range(len(labels)), labels, :]
            # scl_label_loss = self.scl_func(true_label_vectors, labels) / 10

            # center_loss = self.center_loss_fct(anchor_vector, labels)
            # label_distance_loss = self.label_distance_loss_fct(label_vectors)

            loss = ce_loss * self.ce_p + scl_loss * self.scl_p
            # loss = ce_loss * self.ce_p + scl_loss * self.scl_p + scl_label_loss * self.lscl_p
            # loss = ce_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return ZeroShotOutput(
            loss=loss, logits=logits, anchor_vector=anchor_vector, label_vectors=label_vectors,
            hidden_states=sequence_output
        )


