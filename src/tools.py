import numpy as np
import math
from sklearn.metrics import roc_curve, roc_auc_score


def ground_truth_heuristic(X, thresh=0.5):
    shp = X.shape
    N = shp[0]
    M = shp[1]

    if thresh < 1:
        vals = math.floor(M * thresh)
    else:
        vals = thresh

    return np.array([X.sum(axis=1) >= vals], dtype=int)


def get_auc_capped(fpr, tpr, cap=1.0):
    auc = 0
    for i in range(1, len(fpr)):
        width = min(fpr[i],cap) - fpr[i-1]
        height = tpr[i-1]
        auc += width*height
        if fpr[i] > cap: break
    return auc / cap


def get_k_features(mel, k, output='both'):
    coefs = mel.clf_.coef_[0]
    if output != 'bottom':
        topK = np.argpartition(coefs, -k)[-k:]
        topK = topK[np.argsort(coefs[topK])][::-1]
        topK = list(zip(topK, coefs[topK]))
        if output == 'top': return topK
    else:
        bottomK = np.argpartition(coefs, k)[0:k]
        bottomK = bottomK[np.argsort(coefs[bottomK])]
        bottomK = list(zip(bottomK, coefs[bottomK]))
        if output == 'bottom': return bottomK
    return topK, bottomK


def misclassification_error(mel, feats, labels):
    return mel.clf_.score(feats, labels)


def cross_entropy_error(mel, feats, labels):
    res = mel.clf_.predict_log_proba(feats)
    loss = 0.0
    for ind in range(0, len(res)):
        loss += res[ind][labels[ind]]
    return loss / len(labels)


def get_roc(mel, feats, labels):
    probs = mel.clf_.predict_proba(feats)
    fpr, tpr, _ = roc_curve(labels, probs[:,1], pos_label=1)
    auc = roc_auc_score(labels, probs[:,1])
    return fpr, tpr, auc

