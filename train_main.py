from models.zero_shot_text_classify import BertNormalClassification
from transformers import *
from transformers.trainer import TrainingArguments, Trainer
import logging
from utils import container
from utils import metrics
from torch.utils.data import random_split
import os
from datasets.text_classify_dataset import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def dbpedia_main():
    logger.critical("Build Training and validating dataset")
    base_pre_trained_model_path = '/home/ubuntu/likun/nlp_pretrained/bert-google-uncase-base'
    tokenizer = BertTokenizer.from_pretrained(base_pre_trained_model_path)
    dataset_args = container.G({
        "total_data_nums": 5000,

        "train_data_nums": 5000,
        "train_text_path": "/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv",
        "train_use_labels": ['building', 'animal', 'athlete', 'village', 'officeholder',
                             'meanoftransportation', 'plant', 'film', 'writtenwork', 'artist'],
        "train_single_label_max_data_nums": "all",

        "val_percent": 0.1,
        "val_data_nums": 500,
        "val_use_labels": ['naturalplace', 'company', 'album', 'educationalinstitution'],
        "val_single_label_max_data_nums": "all",
        "max_length": 128
    })
    train_dataset = DBpediaConcept(text_path=dataset_args.train_text_path, tokenizer=tokenizer,
                                   max_length=dataset_args.max_length,
                                   data_nums=dataset_args.train_data_nums, use_labels=dataset_args.train_use_labels,
                                   single_label_max_data_nums=dataset_args.train_single_label_max_data_nums)
    train_dataset.build_annotations(train_dataset.nli_data_build)

    val_dataset = DBpediaConcept(text_path=dataset_args.train_text_path, tokenizer=tokenizer,
                                 max_length=dataset_args.max_length,
                                 data_nums=dataset_args.val_data_nums, use_labels=dataset_args.val_use_labels,
                                 single_label_max_data_nums=dataset_args.val_single_label_max_data_nums)
    val_dataset.build_annotations(val_dataset.nli_data_build)

    # train_size = int((1 - dataset_args.val_percent) * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    logger.critical("Build the pre-train model")
    # model_args = container.G({
    #     "num_labels": len(train_dataset.dataset.label2id),
    #     "id2label": train_dataset.dataset.id2label,
    #     "label2id": train_dataset.dataset.label2id,
    # })
    model_args = container.G({
        "num_labels": 2,
        "id2label": {0: 'Not', 1: 'Yes'},
        "label2id": {'Not': 0, 'Yes': 1},
    })
    model = BertForSequenceClassification.from_pretrained(base_pre_trained_model_path, num_labels=model_args.num_labels)

    logger.critical("Setup the training environment")
    training_args = TrainingArguments(
        output_dir='/home/ubuntu/likun/nlp_save_kernels/bert_nil_zero-shot',  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0,  # strength of weight decay
        logging_dir='home/ubuntu/likun/nlp_training_logs/bert_nil_zero-shot',  # directory for storing logs
        logging_steps=10,
        learning_rate=1e-4,
        seed=44,
        no_cuda=False
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=metrics.classify_metrics,
    )

    logger.critical("Start to train and validate")
    trainer.train()
    trainer.evaluate()

    logger.critical("Save the model and config")
    model.config.update({'dataset_args': dataset_args})
    model.config.update({'model_args': model_args})
    trainer.save_model()
    tokenizer.save_vocabulary(training_args.output_dir)

def yahoo_main():
    logger.critical("Build Training and validating dataset")
    base_pre_trained_model_path = '/home/ubuntu/likun/nlp_pretrained/bert-google-uncase-base'
    tokenizer = AutoTokenizer.from_pretrained(base_pre_trained_model_path)
    dataset_args = container.G({
        "total_data_nums": 5000,

        "train_data_nums": 100,
        "train_text_path": "/home/ubuntu/likun/nlp_data/zsl/BenchmarkingZeroShot/topic_yahoo/train_half_0.csv",
        "train_use_labels": "all",
        "train_single_label_max_data_nums": "all",

        "val_data_nums": 100,
        "val_text_path": "/home/ubuntu/likun/nlp_data/zsl/BenchmarkingZeroShot/topic_yahoo/dev.csv",
        "val_use_labels": "all",
        "val_single_label_max_data_nums": "all",

        "max_length": 128,
        "num_labels": 2,
        "id2label": {0: 'Not', 1: 'Yes'},
        "label2id": {'Not': 0, 'Yes': 1},
    })

    train_dataset = TopicYahoo(text_path=dataset_args.train_text_path, tokenizer=tokenizer,
                               max_length=dataset_args.max_length,
                               data_nums=dataset_args.train_data_nums, use_labels=dataset_args.train_use_labels,
                               single_label_max_data_nums=dataset_args.train_single_label_max_data_nums)
    train_dataset.build_annotations(train_dataset.nli_data_build)

    val_dataset = TopicYahoo(text_path=dataset_args.train_text_path, tokenizer=tokenizer,
                             max_length=dataset_args.max_length,
                             data_nums=dataset_args.val_data_nums, use_labels=dataset_args.val_use_labels,
                             single_label_max_data_nums=dataset_args.val_single_label_max_data_nums)
    val_dataset.build_annotations(val_dataset.nli_data_build)

    logger.critical("Build the pre-train model")
    model_args = container.G({
    })

    model = BertNormalClassification.from_pretrained(base_pre_trained_model_path, num_labels=dataset_args.num_labels)

    logger.critical("Setup the training environment")
    training_args = TrainingArguments(
        output_dir='/home/ubuntu/likun/nlp_save_kernels/yahoo_bert_nil_zero-shot',  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0,  # strength of weight decay
        logging_dir='home/ubuntu/likun/nlp_training_logs/yahoo_bert_nil_zero-shot',  # directory for storing logs
        logging_steps=10,
        learning_rate=1e-4,
        seed=44,
        no_cuda=False
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=metrics.classify_metrics,
    )

    logger.critical("Start to train and validate")
    # trainer.train()
    # trainer.evaluate()

    logger.critical("Save the model and config")
    model.config.update({'dataset_args': dataset_args})
    model.config.update({'model_args': model_args})
    trainer.save_model()
    tokenizer.save_vocabulary(training_args.output_dir)


if __name__ == '__main__':
    yahoo_main()