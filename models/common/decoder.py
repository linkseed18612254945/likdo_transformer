from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['TextRNNDecoder']

class TextRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, is_bi=False, dropout=0.1):
        super(TextRNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size,
                            num_layers, batch_first=True, bidirectional=is_bi)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, encoder_feature, seqs, lengths):
        seqs_embedding = self.embedding(seqs)
        # seqs_embedding = self.dropout(seqs_embedding)
        if encoder_feature is not None:
            seqs_embedding = torch.cat((encoder_feature.unsqueeze(1), seqs_embedding), dim=1)  # (batch_size, img_feature + embedding)
        packed = pack_padded_sequence(seqs_embedding, lengths, batch_first=True)
        output, (h, c) = self.rnn(packed)
        outputs, lengths = pad_packed_sequence(output, batch_first=True)
        return output.data, outputs, h

    def sample(self, img_feature, max_length, eos_index=None):
        inputs = img_feature.unsqueeze(1)
        hidden = None
        sampled_ids = []
        for i in range(max_length):
            output, hidden = self.rnn(inputs, hidden)
            output = self.fc(output.squeeze(1))
            _, predicted = output.max(1)
            sampled_ids.append(predicted)
            if eos_index is not None and eos_index == predicted:
                break
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids