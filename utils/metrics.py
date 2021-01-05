from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def classify_metrics(eval_prediction):
    y_true = eval_prediction.label_ids
    y_pred = np.argmax(eval_prediction.predictions, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}