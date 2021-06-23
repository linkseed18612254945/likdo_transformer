from transformers import AutoModelForSequenceClassification,BertTokenizer,RobertaModel, AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizer
from transformers.trainer import TrainingArguments, Trainer
from models.my_trainer import Trainer as MyTrainer
import logging
from utils import metrics
import datasets
import mlflow
import tqdm
import numpy as np
from scipy.spatial import distance
import os
from sklearn.metrics import classification_report
import random
from models import zero_shot_text_classify

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.critical("Build Training and validating dataset")
dataset_args = {
    "dataset_name": "yahoo_topic",
    "data_cache_dir": "/home/ubuntu/likun/huggingface_dataset",
    "train_size": 500,
    "val_size": 30,
    "test_size": 50,
    "max_length": 128,
    "shuffle": True,
}
pre_trained_model_name = 'bert-google-uncase-base'
run_name = "zero-shot-metric-learning-benchmark-topic-medium-changelabel"
# pre_trained_model_name = 'roberta-base'
logger.critical("Build pre-trained model {}".format(pre_trained_model_name))
base_pre_trained_model_path = '/home/ubuntu/likun/nlp_pretrained/{}'.format(pre_trained_model_name)
# trained_model_path = '/home/ubuntu/likun/nlp_save_kernels/zero-shot-metric-learning-benchmark-topic-small'
# tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
tokenizer = BertTokenizer.from_pretrained(base_pre_trained_model_path)


from datasets.features import ClassLabel
from datasets.features import Features
yahoo_zsl_path = '/home/ubuntu/likun/nlp_data/zsl/BenchmarkingZeroShot/topic_yahoo'
fea = Features({
    "text": datasets.Value("string"),
    "label": ClassLabel(names_file=os.path.join(yahoo_zsl_path, 'classes.txt'))
})

download_config = datasets.DownloadConfig()
download_config.max_retries = 20
dataset = datasets.load_dataset('csv', data_files={'train': os.path.join(yahoo_zsl_path, 'train_half_v0.csv'),
                                                   'test': os.path.join(yahoo_zsl_path, 'test.csv')}, features=fea,
                                download_config=download_config, ignore_verifications=True)

if dataset_args['shuffle']:
    dataset = dataset.shuffle()
train_label_index = set(dataset['train']['label'])
fea['label'].names = ['Society Culture', 'Science Mathematics', 'Health', 'Education Reference', 'Computers Internet', 'Sports', 'Business Finance', 'Entertainment Music', 'Family Relationships', 'Politics Government']


test_label_index = set(range(len(fea['label'].names))) - train_label_index

train_label_names = [fea['label'].names[l] for l in train_label_index]
test_label_names = [fea['label'].names[l] for l in test_label_index]

random.shuffle(train_label_names)
random.shuffle(test_label_names)

dataset['test'] = dataset['test'].filter(lambda example: example['label'] in test_label_index)
dataset['train'] = dataset['train'].select(range(dataset_args['train_size'] + dataset_args['val_size'])).map(lambda example: {'label': train_label_names.index(fea['label'].names[example['label']])})
dataset['test'] = dataset['test'].select(range(dataset_args['test_size'])).map(lambda example: {'label': test_label_names.index(fea['label'].names[example['label']])})

train_dataset = dataset['train']
test_dataset = dataset['test']
# label concate process
train_label_names_len = [len(tokenizer.encode(name)) - 2 for name in train_label_names]
def build_labels_position(example):
    lp = []
    end_index = 1
    for nl in train_label_names_len:
        start_index = end_index
        end_index = start_index + nl
        lp.append(tuple(range(start_index, end_index)))
    return lp
train_dataset = train_dataset.map(lambda example: {'label_positions': build_labels_position(example)})
train_dataset = train_dataset.map(lambda example: {'text':  ' '.join(train_label_names) + ' ' + example['text'] })

test_label_names_len = [len(tokenizer.encode(name)) - 2 for name in test_label_names]
def build_labels_position(example):
    lp = []
    end_index = 1
    for nl in test_label_names_len:
        start_index = end_index
        end_index = start_index + nl
        lp.append(tuple(range(start_index, end_index)))
    return lp
test_dataset = test_dataset.map(lambda example: {'label_positions': build_labels_position(example)})
test_dataset = test_dataset.map(lambda example: {'text':  ' '.join(test_label_names) + ' ' + example['text'] })

def standard_encode(examples):
    return tokenizer(examples['text'] , max_length=dataset_args['max_length'], truncation=True, padding='max_length')
train_dataset = train_dataset.map(standard_encode, batched=True)
test_dataset = test_dataset.map(standard_encode, batched=True)
val_dataset = train_dataset.select(range(dataset_args['train_size'], dataset_args['train_size'] + dataset_args['val_size']))
train_dataset = train_dataset.select(range(dataset_args['train_size']))

logger.critical("Setup the training environment")
model = zero_shot_text_classify.BertMetricLearningModel.from_pretrained(base_pre_trained_model_path)
# model = BertMetricLearningModel.from_pretrained(trained_model_path)
model.config.return_dict = True

num_train_epochs = 3
train_batch_size = 32
warmup_steps = int(len(train_dataset) * num_train_epochs // train_batch_size * 0.1)
training_args = TrainingArguments(
    output_dir='/home/ubuntu/likun/nlp_save_kernels/{}'.format('rrr'),  # output directory
    num_train_epochs=num_train_epochs,  # total number of training epochs
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
    weight_decay=0.99,  # strength of weight decay
    logging_dir='/home/ubuntu/likun/nlp_training_logs/{}'.format('rrr'),  # directory for storing logs
    logging_steps=10,
    learning_rate=5e-5,
    seed=49,
    no_cuda=False,
)

train_params = {k: v for k, v in training_args.__dict__.items() if (isinstance(v, int) or isinstance(v, float)) and not isinstance(v, bool)}
trainer = MyTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    data_collator=zero_shot_text_classify.zero_shot_data_collator,
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=None,  # evaluation dataset
    compute_metrics=None,
)

def iter_test_eval(test_dataset, label_names, chunk_size=512):
    start_index = 0
    end_index = 0
    y_predict = []
    y_true = []
    all_text_vectors = []
    all_label_vectors = []
    while end_index < len(test_dataset):
        start_index = end_index
        end_index = min(end_index + chunk_size, len(test_dataset))
        select_dataset = test_dataset.select(range(start_index, end_index))
        predict_dataset = select_dataset.map(remove_columns=['label'])
        predict_res = trainer.predict(predict_dataset)
        y_predict.extend(np.argmax(predict_res.predictions, axis=1))
        all_text_vectors.extend(predict_res.extra_info['anchor_vector'].cpu().numpy())
        all_label_vectors.extend(predict_res.extra_info['label_vectors'].cpu().numpy())
        print("Process end index: {}".format(end_index))
    y_true = test_dataset['label']
    mres = {'eval_{}'.format(k): v for k, v in metrics.base_classify_metrics(y_true, y_predict).items()}
    all_label_vectors = np.array(all_label_vectors)
    label_res_vectors = []
    for label in range(len(label_names)):
        label_row_index = [i for i, l in enumerate(zip(y_predict, y_true)) if label == l[0] == l[1]]
        label_vector = np.average(all_label_vectors[label_row_index, label, :], axis=0)
#         label_vector = all_label_vectors[label_row_index[0], label, :]
        label_res_vectors.append(label_vector)
    all_text_vectors = np.array(all_text_vectors).squeeze(1)
    label_res_vectors = np.array(label_res_vectors)
    return y_true, y_predict, mres, all_text_vectors, label_res_vectors, all_label_vectors

def iter_eval(test_dataset, label_names, chunk_size=512):
    start_index = 0
    end_index = 0
    y_predict = []
    y_true = []
    all_text_vectors = []
    all_label_vectors = []
    while end_index < len(test_dataset):
        start_index = end_index
        end_index = min(end_index + chunk_size, len(test_dataset))
        select_dataset = test_dataset.select(range(start_index, end_index))
        predict_dataset = select_dataset.map(remove_columns=['label'])
        predict_res = trainer.predict(predict_dataset)
        y_predict.extend(np.argmax(predict_res.predictions, axis=1))
        all_text_vectors.extend(predict_res.extra_info['anchor_vector'].cpu().numpy())
        all_label_vectors.extend(predict_res.extra_info['label_vectors'].cpu().numpy())
        print("Process end index: {}".format(end_index))
    y_true = test_dataset['label']
    mres = {'eval_{}'.format(k): v for k, v in metrics.base_classify_metrics(y_true, y_predict).items()}
    return y_true, y_predict, mres, None, None, None

train_res = trainer.train(test_func=iter_eval, test_data=val_dataset, test_label_names=train_label_names)