from transformers import pipelines
from transformers import BartTokenizer, BartModel
import numpy as np

class NLIArgumentHandler(pipelines.ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs


class MyZeroShotClassificationPipeline(pipelines.Pipeline):
    def __init__(self, args_parser=NLIArgumentHandler(), *args, **kwargs):
        super().__init__(*args, args_parser=args_parser, **kwargs)

    def _parse_and_tokenize(self, *args, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
            truncation="only_first",
        )

        return inputs

    def __call__(self, sequences, candidate_labels, hypothesis_template="This example is {}.", multi_class=False):
        outputs = super().__call__(sequences, candidate_labels, hypothesis_template)
        num_sequences = 1 if isinstance(sequences, str) else len(sequences)
        candidate_labels = self._args_parser._parse_labels(candidate_labels)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

        if len(candidate_labels) == 1:
            multi_class = True

        if not multi_class:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., -1]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            entail_contr_logits = reshaped_outputs[..., [0, -1]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]

        result = []
        for iseq in range(num_sequences):
            top_inds = list(reversed(scores[iseq].argsort()))
            result.append(
                {
                    "sequence": sequences if isinstance(sequences, str) else sequences[iseq],
                    "labels": [candidate_labels[i] for i in top_inds],
                    "scores": scores[iseq][top_inds].tolist(),
                }
            )

        if len(result) == 1:
            return result[0]
        return result

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained("/home/ubuntu/likun/nlp_pretrained/bart-large-mnli")
    model = BartModel.from_pretrained("/home/ubuntu/likun/nlp_pretrained/bart-large-mnli")
    candidate_labels = ['political', 'sport']
    text = 'Obama like china'
    res = tokenizer(text)
    # classifier = MyZeroShotClassificationPipeline(model="/home/ubuntu/likun/nlp_pretrained/bart-large-mnli")
    n_classifier = pipelines.pipeline("zero-shot-classification", model="/home/ubuntu/likun/nlp_pretrained/bart-large-mnli")
    print(n_classifier(text, candidate_labels))