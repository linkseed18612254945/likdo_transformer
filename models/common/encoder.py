

import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['ImageCNNEncoder', 'TextRNNEncoder']

class ImageCNNEncoder(nn.Module):
    def __init__(self, image_embedding_dim):
        super(ImageCNNEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, image_embedding_dim)
        self.bn = nn.BatchNorm1d(image_embedding_dim, momentum=0.01)

    def forward(self, img):
        with torch.no_grad():
            feature = self.resnet(img)
        feature = feature.reshape(feature.size(0), -1)
        feature = self.fc(feature)
        feature = self.bn(feature)
        return feature


class TextRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, is_bi=False, dropout=0.1):
        super(TextRNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size,
                           num_layers, batch_first=True, bidirectional=is_bi)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seqs):
        seqs_embedding = self.embedding(seqs)
        seqs_embedding = self.dropout(seqs_embedding)
        outputs, (h, c) = self.rnn(seqs_embedding)
        return outputs, h


class TextCNN(nn.Module):
    def __init__(self, input_size, embedding_size, output_size, filter_sizes=(1, 2, 3, 4), num_filters=3, pooling_method='max'):
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.pooling_method = pooling_method
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, kernel_size=(filter_size, embedding_size)) for filter_size in filter_sizes])
        self.activate = nn.ReLU()
        self.linear = nn.Linear(len(filter_sizes) * num_filters, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        sequence_length = embedded.shape[1]
        conv_pooling_res = []
        for conv in self.convs:
            conved = conv(embedded.unsqueeze(dim=1))
            conved = self.activate(conved)
            if self.pooling_method == 'max':
                pooling = nn.MaxPool2d(kernel_size=(sequence_length - conv.kernel_size[0] + 1, 1))
            else:
                pooling = nn.AvgPool2d(kernel_size=(sequence_length - conv.kernel_size[0] + 1, 1))
            pooled = pooling(conved)
            conv_pooling_res.append(pooled)

        output = torch.cat(conv_pooling_res, dim=3)
        output = torch.reshape(output, shape=(-1, len(self.filter_sizes) * self.num_filters))
        output = self.linear(output)
        return output