
from sklearn.metrics import roc_auc_score as ras
import numpy as np

def AUC(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc = ras(y_true,y_pred)
    return auc