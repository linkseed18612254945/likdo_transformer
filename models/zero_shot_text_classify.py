from transformers import BertPreTrainedModel, BertModel
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

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
        self.scl_t = 0.1

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
        :param anchor_vector:
        :param labels:
        :return:
        """

        total_losses = 0
        for i in range(anchor_vectors.shape[0]):
            anchor_vector = anchor_vectors[i, :]
            other_index = torch.from_numpy(np.tile(np.array(list(filter(lambda x: x != 1, anchor_vectors.shape[0]))),
                                                   anchor_vectors.shape[1]).reshape(anchor_vectors.shape[1], -1))
            # other_vectors = np.delete(anchor_vectors, i, 0)
            other_vectors = torch.gather(anchor_vectors.transpose(1, 0), dim=1, index=other_index).transpose(1, 0)
            same_labels = torch.where(other_vectors == labels[i])
            same_label_vectors = anchor_vectors[same_labels]
            singe_sample_loss = torch.sum(torch.log(torch.exp(torch.matmul(same_label_vectors, anchor_vector) / self.scl_t) / torch.sum(torch.exp(torch.matmul(other_vectors, anchor_vector) / self.scl_t)))) / (anchor_vectors.shape[0] - 1)
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
        # attentions = torch.mean(outputs.attentions[-1], dim=1)
        # label_attentions = None
        # for positions in label_positions:
        #     x = None
        #     for j in range(positions.shape[1]):
        #         label_attention = attentions[range(positions.shape[0]), :, positions[:, j]].unsqueeze(dim=1)
        #         if x is None:
        #             x = label_attention
        #         else:
        #             x = torch.cat([x, label_attention], dim=1)
        #     label_attention = torch.mean(x, dim=1).unsqueeze(dim=1)
        #     if label_attentions is None:
        #         label_attentions = label_attention
        #     else:
        #         label_attentions = torch.cat([label_attentions, label_attention], dim=1)

        anchor_vector = sequence_output[:, 0, :].unsqueeze(dim=1)
        anchor_vector = self.metric_linear(anchor_vector)
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

        # for positions in label_positions:
        #     x = None
        #     for j in range(positions.shape[1]):
        #         label_vector = sequence_output[range(positions.shape[0]), positions[:, j], :].unsqueeze(dim=1)
        #         if x is None:
        #             x = label_vector
        #         else:
        #             x = torch.cat([x, label_vector], dim=1)
        #     label_vector = torch.mean(x, dim=1).unsqueeze(dim=1)
        #     if label_vectors is None:
        #         label_vectors = label_vector
        #     else:
        #         label_vectors = torch.cat([label_vectors, label_vector], dim=1)


        label_vectors = self.dropout(label_vectors)
        label_vectors = self.metric_linear(label_vectors)
        logits = torch.cosine_similarity(label_vectors, anchor_vector, dim=2)

        loss = None
        if labels is not None:
            ce_loss = self.ce_loss_fct(logits, labels)
            # center_loss = self.center_loss_fct(anchor_vector, labels)
            # label_distance_loss = self.label_distance_loss_fct(label_vectors)
            # loss = ce_loss + label_distance_loss
            loss = ce_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return ZeroShotOutput(
            loss=loss, logits=logits, anchor_vector=anchor_vector, label_vectors=label_vectors, hidden_states=outputs.hidden_states, attentions=attentions,
        )


