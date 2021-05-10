from transformers import RobertaForMaskedLM, RobertaTokenizerFast, AutoTo
from transformers.trainer import TrainingArguments, Trainer
import logging
from utils import metrics
import datasets
import mlflow
from utils import container


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    pre_trained_model_name = 'bert-google-uncase-base'
    logger.critical("Build pre-trained model {}".format(pre_trained_model_name))
    base_pre_trained_model_path = '/home/ubuntu/likun/nlp_pretrained/{}'.format(pre_trained_model_name)
    tokenizer = RobertaTokenizer.from_pretrained(base_pre_trained_model_path)
    logger.critical("Build Training and validating dataset")
    dataset_args = {
        "name": "ag_news",
        "train_size": 10000,
        "val_size": 3000,
        "test_size": 1000,
        "max_length": 128
    }

    def encode(examples):
        return tokenizer(examples['text'], max_length=dataset_args['max_length'], truncation=True, padding='max_length')

    dataset = datasets.load_dataset(dataset_args['name'])
    train_dataset = dataset['train'].train_test_split(train_size=dataset_args['train_size'],
                                                      test_size=dataset_args['val_size'])
    train_dataset, val_dataset = train_dataset['train'], train_dataset['test']
    test_dataset = dataset['test'].train_test_split(train_size=dataset_args['test_size'])
    test_dataset = test_dataset['train']
    mlflow.log_params()
    train_dataset = train_dataset.map(encode, batched=True)
    val_dataset = val_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)
    tokenizer.get_vocab()
    logger.critical("Setup the training environment")
    model = RobertaForMaskedLM.from_pretrained(base_pre_trained_model_path,
                                                               num_labels=train_dataset.features['label'].num_classes,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
    model.config.return_dict = True
    training_args = TrainingArguments(
        output_dir='/home/ubuntu/likun/nlp_save_kernels/bert_agnews',  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=0,  # number of warmup steps for learning rate scheduler
        weight_decay=0,  # strength of weight decay
        logging_dir='home/ubuntu/likun/nlp_training_logs/bert_agnews',  # directory for storing logs
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

    logger.critical("Start to train and validate and test")
    trainer.train()
    trainer.evaluate()


    res = trainer.predict(test_dataset)
    container.G({
        "label_ids": '',
        "predictions": res.label_ids
    })

    logger.critical("Save the model and config")
    trainer.save_model()
    tokenizer.save_vocabulary(training_args.output_dir)

if __name__ == '__main__':
    main()