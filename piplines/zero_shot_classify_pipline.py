import tqdm
# from utils.logger import get_logger
from transformers import pipeline
from sklearn.metrics import classification_report
# logger = get_logger(__file__)
import pickle


def zero_shot_text_classify_main():
    # logger.critical("Build test/predict config and device")
    # df = pd.read_csv('/home/ubuntu/likun/nlp_data/zsl/BenchmarkingZeroShot/situation_st/test.csv')
    # df = df.reset_index(drop=True)

    data_path = '/home/ubuntu/likun/likdo_transformer/situation_st_pickle_data_all_v1.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    test_label_names = data['test_label_names']
    data = data['test_dataset']
    labels = data['label']
    texts = data['text']
    texts = [t[90:] for t in texts]
    classifier = pipeline("zero-shot-classification",device=1)
    predict_labels = []
    for text in tqdm.tqdm(texts, desc='Zero-shot predicting'):
        res = classifier(text, test_label_names)
        predict_labels.append(test_label_names.index(res['labels'][0]))
    with open('/home/ubuntu/likun/likdo_transformer/zsl_pipline_situation_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    report = classification_report(labels, predict_labels, target_names=test_label_names)
    print(report)


if __name__ == '__main__':
    zero_shot_text_classify_main()
