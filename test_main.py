from transformers import pipeline
from utils import container
from sklearn.metrics import classification_report
from datasets.text_classify_dataset import *
import tqdm

dataset_args = container.G({
        "predict_data_nums": 3000,
        "predict_text_path": "/home/ubuntu/likun/nlp_data/zsl/BenchmarkingZeroShot/topic_yahoo/test.csv",
        "predict_use_labels": "all",
        "predict_single_label_max_data_nums": "all",
        "max_length": 128
    })
predict_dataset = TopicYahoo(text_path=dataset_args.predict_text_path, tokenizer=None,
                             max_length=dataset_args.max_length,
                             data_nums=dataset_args.predict_data_nums, use_labels=dataset_args.predict_use_labels,
                             single_label_max_data_nums=dataset_args.predict_single_label_max_data_nums)
predict_dataset.build_annotations(predict_dataset.classify_data_build)
classifier = pipeline('zero-shot-classification', model='/home/ubuntu/likun/nlp_pretrained/bart-large-mnli', device=1)
predict_labels = []
true_labels = []
for index, item in tqdm.tqdm(predict_dataset.annotations.items(), desc="Testing"):
    res = classifier(item['text'], TopicYahoo.labels)
    true_labels.append(item['label_name'])
    predict_labels.append(res['labels'][0].lower())
report = classification_report(true_labels, predict_labels)
print(report)


