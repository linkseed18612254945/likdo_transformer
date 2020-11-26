import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics

def bi_evaluate(y_true, y_pred):
    tp = np.sum(np.multiply(y_true, y_pred))
    fp = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    fn = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    tn = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

    acc = (tp + tn) / (tp + tn + fp + fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 / ((1 / p) + (1 /r))
    return tp, fp, fn, tn, acc, p, r, f1

def multi_evaluate_report(y_true, y_pred, labels=None, target_names=None):
    return classification_report(y_true, y_pred, labels, target_names)