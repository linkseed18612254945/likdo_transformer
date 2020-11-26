import nltk
from torch.utils.data import dataset
from utils import io
from datasets.common.vocab import Vocab, build_vocab
from datasets.common.transforms import *
from torch.utils.data import dataloader, random_split
from utils.logger import get_logger
from utils.text_utils import pad_sentences_batch
import tqdm
from collections import defaultdict
logger = get_logger(__file__)

class TextClassifyDataset(dataset.Dataset):
    def __init__(self, text_path, tokenizer, data_nums='all', use_labels='all', single_label_max_data_nums='all'):
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
        self.annotation = self.get_annotations()

    def get_annotations(self):
        """
        Compile the data file and return a {index: row} data
        :return:
        """
        raise NotImplementedError

    def __getitem__(self, index):
        item = self.annotation[index]

        item = {key: torch.tensor(val) for key, val in item}

        if self.text_transformer is not None:
            text = self.text_transformer((self.annotations[index]['text']))
        else:
            text = self.vocab.map_sequence(self.annotations[index]['text'])

        if self.item_transformer is not None:
            item = self.item_transformer((self.annotations[index]['item']))
        else:
            item = self.vocab.map_sequence(self.annotations[index]['item'])

        if self.label_transformer is not None:
            label = self.label_transformer([self.annotations[index]['label']])
        else:
            label = self.annotations[index]['label']

        if self.label_name_transformer is not None:
            label_name = self.label_name_transformer((self.annotations[index]['label_name']))
        else:
            label_name = self.vocab.map_sequence(self.annotations[index]['label_name'])

        return text, item, label, label_name

    def __len__(self):
        return len(self.annotations.keys())




class DBpediaConcept(dataset.Dataset):
    def __init__(self, text_path, text_vocab_json, label_vocab_json, text_transformer=None,
                 item_transformer=None, label_transformer=None, label_name_transformer=None,
                 data_nums='all', use_labels='all', single_label_max_data_nums='all'):
        super(DBpediaConcept, self).__init__()

        # initialize args
        self.text_path = text_path
        self.vocab = Vocab.from_json(text_vocab_json)
        self.label_vocab = Vocab.from_json(label_vocab_json)
        self.data_nums = data_nums
        self.num_labels = len(self.label_vocab.idx2word)
        self.use_labels = use_labels
        self.single_label_max_data_nums = single_label_max_data_nums
        self.label_counts = defaultdict(int)
        self.annotations = self.get_annotations()

        self.text_transformer = text_transformer
        self.item_transformer = item_transformer
        self.label_transformer = label_transformer
        self.label_name_transformer = label_name_transformer

    def get_annotations(self):
        annotations = defaultdict(dict)
        df = io.load_csv(self.text_path, without_header=False, shuffle=True)
        data_nums = df.shape[0] if self.data_nums == 'all' else self.data_nums
        annotation_index = 0
        for _, row in tqdm.tqdm(df.iterrows(), desc='Build dataset annotations', total=data_nums):
            label_name = row['label'].lower()
            if self.use_labels != 'all' and label_name not in self.use_labels:
                continue
            if self.single_label_max_data_nums != 'all' and self.label_counts[label_name] > self.single_label_max_data_nums:
                continue
            annotations[annotation_index] = {'label': self.label_vocab.word2idx[label_name],
                                             'label_name': label_name,
                                             'text': row['text'].lower(),
                                             'item': row['item'].lower()}
            self.label_counts[label_name] += 1
            annotation_index += 1
            if annotation_index >= data_nums:
                break
        return annotations

    def __getitem__(self, index):
        if self.text_transformer is not None:
            text = self.text_transformer((self.annotations[index]['text']))
        else:
            text = self.vocab.map_sequence(self.annotations[index]['text'])

        if self.item_transformer is not None:
            item = self.item_transformer((self.annotations[index]['item']))
        else:
            item = self.vocab.map_sequence(self.annotations[index]['item'])

        if self.label_transformer is not None:
            label = self.label_transformer([self.annotations[index]['label']])
        else:
            label = self.annotations[index]['label']

        if self.label_name_transformer is not None:
            label_name = self.label_name_transformer((self.annotations[index]['label_name']))
        else:
            label_name = self.vocab.map_sequence(self.annotations[index]['label_name'])

        return text, item, label, label_name

    def __len__(self):
        return len(self.annotations.keys())

    @classmethod
    def collate_fn(cls, data):
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[0]), reverse=True)
        texts, items, labels, label_names = zip(*data)
        if isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, 0).squeeze()

        text_targets, text_lengths = pad_sentences_batch(texts)
        item_targets, items_lengths = pad_sentences_batch(items)
        label_name_targets, label_name_lengths = pad_sentences_batch(label_names)

        feed_dict = {
            "texts": text_targets,
            "text_lengths": text_lengths,
            "items": item_targets,
            "labels": labels,
            "label_names": label_name_targets,
            "is_static_vector": False
        }
        return feed_dict

def get_train_data(config, text_transformer=None, label_transformer=torch.LongTensor, label_name_transformer=None, collate_fn='default'):
    if 'dbpedia' in config.data.name:
        train_dataset = DBpediaConcept(text_path=config.data.train_text_path,
                                       text_vocab_json=config.data.vocab_path,
                                       label_vocab_json=config.data.label_vocab_path,
                                       text_transformer=text_transformer, label_transformer=label_transformer,
                                       label_name_transformer=label_name_transformer,
                                       data_nums=config.data.train_data_nums, use_labels=config.data.train_use_labels,
                                       single_label_max_data_nums=config.data.train_single_label_max_data_nums)
        if config.data.valid_text_path is not None:
            valid_dataset = DBpediaConcept(text_path=config.data.valid_text_path,
                                           text_vocab_json=config.data.vocab_path,
                                           label_vocab_json=config.data.label_vocab_path,
                                           text_transformer=text_transformer, label_transformer=label_transformer,
                                           label_name_transformer=label_name_transformer,
                                           data_nums=config.data.valid_data_nums,
                                           use_labels=config.data.valid_use_labels,
                                           single_label_max_data_nums=config.data.valid_single_label_max_data_nums)
        else:
            train_size = int((1 - config.train.valid_percent) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])

    elif 'news20' in config.data.name:
        train_dataset = News20(text_path=config.data.train_text_path,
                               text_vocab_json=config.data.vocab_path,
                               label_vocab_json=config.data.label_vocab_path,
                               text_transformer=text_transformer, label_transformer=label_transformer,
                               label_name_transformer=label_name_transformer,
                               data_nums=config.data.train_data_nums)
        if config.data.valid_text_path is not None:
            valid_dataset = News20(text_path=config.data.valid_text_path,
                                   text_vocab_json=config.data.vocab_path,
                                   label_vocab_json=config.data.label_vocab_path,
                                   text_transformer=text_transformer, label_transformer=label_transformer,
                                   label_name_transformer=label_name_transformer, data_nums=config.data.train_data_nums)
        else:
            train_size = int((1 - config.train.valid_percent) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        raise Exception(f'Not support data {config.data.name}')

    train_dataset = train_dataset.dataset if isinstance(train_dataset, dataset.Subset) else train_dataset
    valid_dataset = valid_dataset.dataset if isinstance(valid_dataset, dataset.Subset) else valid_dataset

    collate_fn = train_dataset.collate_fn if collate_fn == 'default' else collate_fn
    train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                                         shuffle=config.train.shuffle, collate_fn=collate_fn)
    valid_loader = dataloader.DataLoader(dataset=valid_dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                                         shuffle=config.train.shuffle, collate_fn=collate_fn)
    return train_dataset, train_loader, valid_dataset, valid_loader


def get_test_data(config, text_transformer=None, label_transformer=torch.LongTensor, label_name_transformer=None, collate_fn='default'):
    if 'dbpedia' in config.data.name:
        test_dataset = DBpediaConcept(text_path=config.data.test_text_path,
                                      text_vocab_json=config.data.vocab_path,
                                      label_vocab_json=config.data.label_vocab_path,
                                      text_transformer=text_transformer, label_transformer=label_transformer,
                                      label_name_transformer=label_name_transformer,
                                      data_nums=config.data.test_data_nums, use_labels=config.data.test_use_labels,
                                      single_label_max_data_nums=config.data.test_single_label_max_data_nums)

    elif 'news20' in config.data.name:
        test_dataset = News20(text_path=config.data.test_text_path,
                              text_vocab_json=config.data.vocab_path,
                              label_vocab_json=config.data.label_vocab_path,
                              text_transformer=text_transformer, label_transformer=label_transformer,
                              label_name_transformer=label_name_transformer,
                              data_nums=config.data.test_data_nums, use_labels=config.data.test_use_labels,
                              single_label_max_data_nums=config.data.test_single_label_max_data_nums)
    else:
        raise Exception(f'Not support data {config.data.name}')
    return test_dataset

if __name__ == '__main__':
    data_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'
    text_vocab_path = '/home/ubuntu/likun/vocab/news20_vocab.json'
    label_vocab_path = '/home/ubuntu/likun/vocab/dbpedia_label_vocab.json'

    # # build vocab
    build_vocab(file_paths=(data_path,),
                compile_functions=(DBpediaConcept.get_label_name_from_csv,),
                extra_param=None, vocab_path=label_vocab_path, add_special_token=False)