from transformers import PreTrainedModel


class RNNBaseLine(PreTrainedModel):
    def __init__(self, config):
        super(RNNBaseLine, self).__init__(config)

    def forward(self):
        pass