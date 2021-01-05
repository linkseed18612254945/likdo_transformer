from transformers import pipeline
from utils import container
import pandas as pd
from datasets.text_classify_dataset import DBpediaConcept

for index, item in predict_dataset.annotations.items():
    print(item['label_name'])
    text = f"{item['item']} is a [MASK] category"
    res = unmasker(text)
    print(res)
def main():
    dataset_args = container.G({
        "predict_data_nums": 10,
        "predict_text_path": "/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv",
        "predict_use_labels": "all",
        "predict_single_label_max_data_nums": "all",
        "max_length": 128
    })
    predict_dataset = DBpediaConcept(text_path=dataset_args.predict_text_path, tokenizer=None,
                                     max_length=dataset_args.max_length,
                                     data_nums=dataset_args.predict_data_nums, use_labels=dataset_args.predict_use_labels,
                                     single_label_max_data_nums=dataset_args.predict_single_label_max_data_nums)
    predict_dataset.build_annotations(predict_dataset.classify_data_build)
    unmasker = pipeline('fill-mask', model='/home/ubuntu/likun/nlp_pretrained/bert-google-uncase-base')
    for index, item in predict_dataset.annotations.items():
        print(item['label_name'])
        text = f"{item['item']} is a [MASK] category"
        res = unmasker(text)
        print(res)
if __name__ == '__main__':
    main()