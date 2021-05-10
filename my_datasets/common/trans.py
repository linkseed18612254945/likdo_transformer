import torchvision
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel

class GloveVectorTransform(object):
    def __init__(self, glove_file_path):
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            glove_vectors = f.read().splitlines()
        self.glove_vectors = {l.split()[0]: l.split()[1:] for l in glove_vectors}

    def __call__(self, sentence):
        return torch.FloatTensor([list(map(float, self.glove_vectors.get(word))) for word in sentence])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BertTokenizerTransformer(object):
    def __init__(self, bert_base_path):

        self.tokenizer = BertTokenizer.from_pretrained(bert_base_path, do_lower_case=True)

    def __call__(self, sentence):
        input_ids = self.tokenizer.encode(
            sentence,
            truncation=True,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=50,  # 设定最大文本长度
            pad_to_max_length=True,  # pad到最大的长度
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        return input_ids.squeeze()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SentenceBertTransformer(object):
    def __init__(self, bert_base_path):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_base_path)
        self.model = AutoModel.from_pretrained(bert_base_path)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def __call__(self, sentence):
        sentences = [sentence]
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.squeeze()

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_image_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
    return transform


def standard_encode(examples, tokenizer):
    tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')