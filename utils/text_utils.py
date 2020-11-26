import torch
import pandas as pd
import random


def label_name_random_split(label_names, first_group_num):
    assert first_group_num <= len(label_names), "First group num should less than label type's num"
    first_group = random.sample(label_names, first_group_num)
    last_group = [label for label in label_names if label not in first_group]
    return first_group, last_group


def pad_sentences_batch(sentences_batch):
    lengths = [len(sentence) for sentence in sentences_batch]
    targets = torch.zeros(len(sentences_batch), max(lengths)).long()
    for i, sentence in enumerate(sentences_batch):
        sentence = torch.LongTensor(sentence)
        end = lengths[i]
        targets[i, :end] = sentence[:end]
    return targets, lengths

def multi_label_flat(df, label_col, split_seg=' '):
    res = {col: [] for col in df.columns if col != label_col}
    label_res = []
    for index, row in df.iterrows():
        labels = row[label_col].split(split_seg)
        for label in labels:
            label_res.append(label)
            for col in res:
                res[col].append(row[col])
    res[label_col] = label_res
    res_df = pd.DataFrame(data=res)
    return res_df

