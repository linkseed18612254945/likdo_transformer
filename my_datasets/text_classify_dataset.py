from torch.utils.data import dataset
from utils import io
import torch
from utils.logger import get_logger
import tqdm
from collections import defaultdict
from transformers import BatchEncoding
import random
logger = get_logger(__file__)

class TextClassifyDataset(dataset.Dataset):
    def __init__(self, text_path, tokenizer, id2label=None, max_length=128, data_nums='all', use_labels='all', single_label_max_data_nums='all'):
        """
        :param text_path:  Dataset file path, normally is a csv format file
        :param tokenizer:  Transformer.PretrainTokenizer object, can transform text to id list and attention mask
        :param data_nums:  Total use data nums
        :param use_labels: Use label
        :param single_label_max_data_nums: Per label max use nums
        """
        super(TextClassifyDataset, self).__init__()
        self.text_path = text_path
        self.tokenizer = tokenizer
        self.data_nums = data_nums
        self.use_labels = use_labels
        self.single_label_max_data_nums = single_label_max_data_nums
        self.id2label = id2label
        self.label2id = None
        self.max_length = max_length
        self.annotations = None

    def build_annotations(self, data_build_func):
        """
        Compile the data file and return a {index: row} data
        :return:
        """
        raise NotImplementedError

    def encode_fn(self, val):
        if isinstance(val, int):
            return torch.LongTensor([val])
        input_ids = self.tokenizer(
            val,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            pad_to_max_length=True,  # pad到最大的长度
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        return input_ids

    def __getitem__(self, index):
        item = self.annotations[index]
        item_inputs = {}
        for key, val in item.items():
            encode_res = self.encode_fn(val)
            if isinstance(encode_res, BatchEncoding):
                for k, v in encode_res.items():
                    if key == 'text':
                        item_inputs[k] = v.squeeze(0)
                    else:
                        item_inputs[f'{key}_{k}'] = v.squeeze(0)
            else:
                item_inputs[key] = encode_res
        return item_inputs

    def __len__(self):
        return len(self.annotations.keys())

class TopicYahoo(TextClassifyDataset):
    labels = ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
              'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music',
              'Family & Relationships', 'Politics & Government']

    def __init__(self, text_path, tokenizer, label_map=None, max_length=128, data_nums='all', use_labels='all',
                 single_label_max_data_nums='all'):
        super(TopicYahoo, self).__init__(text_path, tokenizer, label_map, max_length, data_nums,
                                         use_labels, single_label_max_data_nums)

    def build_annotations(self, data_build_func):
        if data_build_func is None:
            data_build_func = self.classify_data_build
        label_counts = defaultdict(int)
        annotations = defaultdict(dict)
        df = io.load_csv(self.text_path, without_header=False, shuffle=True)
        self.id2label = {i: l.lower() for i, l in enumerate(TopicYahoo.labels)} if self.id2label is None else self.id2label
        self.label2id = {l: i for i, l in self.id2label.items()}
        data_nums = df.shape[0] if self.data_nums == 'all' else self.data_nums
        annotation_index = 0
        for _, row in tqdm.tqdm(df.iterrows(), desc='Build Topic Yahoo dataset annotations', total=data_nums):
            label_name = row['label'].lower()
            if self.use_labels != 'all' and label_name not in self.use_labels:
                continue
            if self.single_label_max_data_nums != 'all' and label_counts[label_name] > self.single_label_max_data_nums:
                continue
            build_data = data_build_func(row)
            for bd in build_data:
                annotations[annotation_index] = bd
                label_counts[label_name] += 1
                annotation_index += 1
            if annotation_index >= data_nums:
                break
        self.annotations = annotations

    def classify_data_build(self, row):
        label_name = row['label'].lower()
        return [{'label': self.label2id[label_name], 'label_name': label_name,
                'text': row['text'].lower()}]

    def nli_data_build(self, row):
        bds = []
        nli_query_formats = "[SEP] this text is about {} ?"
        positive_hypothesis = row['text'].lower() + nli_query_formats.format(row['label'].lower())
        bds.append({'text': positive_hypothesis, 'label': 1})
        use_labels = [l for l in self.label2id.keys() if l != row['label']] if self.use_labels == 'all' \
            else [l for l in self.use_labels if l != row['label']]

        negative_hypothesises = {'text': row['text'].lower() + nli_query_formats.format(random.choice(use_labels)),
                                 'label': 0}
        bds.append(negative_hypothesises)
        # negative_hypothesises = [row['text'].lower() + nli_query_formats.format(l) for l in use_labels]
        # bds.extend([{'text': nh, 'label': 0} for nh in negative_hypothesises])
        return bds

class DBpediaConcept(TextClassifyDataset):
    labels = ['building', 'animal', 'athlete', 'village', 'officeholder', 'meanoftransportation',
              'plant', 'film', 'writtenwork', 'artist', 'naturalplace', 'company', 'album', 'educationalinstitution']

    def __init__(self, text_path, tokenizer, label_map=None, max_length=128, data_nums='all', use_labels='all',
                 single_label_max_data_nums='all'):
        super(DBpediaConcept, self).__init__(text_path, tokenizer, label_map, max_length, data_nums,
                                             use_labels, single_label_max_data_nums)

    def build_annotations(self, data_build_func):
        if data_build_func is None:
            data_build_func = self.classify_data_build
        label_counts = defaultdict(int)
        annotations = defaultdict(dict)
        df = io.load_csv(self.text_path, without_header=False, shuffle=True)
        self.id2label = {i: l for i, l in enumerate(set(df['label'].apply(lambda x: x.lower())))} if self.id2label is None else self.id2label
        self.label2id = {l: i for i, l in self.id2label.items()}
        data_nums = df.shape[0] if self.data_nums == 'all' else self.data_nums
        annotation_index = 0
        for _, row in tqdm.tqdm(df.iterrows(), desc='Build DBpedia dataset annotations', total=data_nums):
            label_name = row['label'].lower()
            if self.use_labels != 'all' and label_name not in self.use_labels:
                continue
            if self.single_label_max_data_nums != 'all' and label_counts[label_name] > self.single_label_max_data_nums:
                continue
            build_data = data_build_func(row)
            for bd in build_data:
                annotations[annotation_index] = bd
                label_counts[label_name] += 1
                annotation_index += 1
            if annotation_index >= data_nums:
                break
        self.annotations = annotations

    def classify_data_build(self, row):
        label_name = row['label'].lower()
        return [{'label': self.label2id[label_name], 'label_name': label_name,
                'text': row['text'].lower(), 'item': row['item'].lower()}]

    def nli_data_build(self, row):
        bds = []
        nli_query_formats = "[SEP] this text is about {} ?"
        positive_hypothesis = row['text'].lower() + nli_query_formats.format(row['label'].lower())
        bds.append({'text': positive_hypothesis, 'label': 1})
        use_labels = [l for l in self.label2id.keys() if l != row['label']] if self.use_labels == 'all' \
            else [l for l in self.use_labels if l != row['label']]

        negative_hypothesises = {'text': row['text'].lower() + nli_query_formats.format(random.choice(use_labels)), 'label':0}
        bds.append(negative_hypothesises)
        # negative_hypothesises = [row['text'].lower() + nli_query_formats.format(l) for l in use_labels]
        # bds.extend([{'text': nh, 'label': 0} for nh in negative_hypothesises])
        return bds


if __name__ == '__main__':
    data_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'
    text_vocab_path = '/home/ubuntu/likun/vocab/news20_vocab.json'
    label_vocab_path = '/home/ubuntu/likun/vocab/dbpedia_label_vocab.json'

