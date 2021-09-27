# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:01:40 2020

@author: 00098223
"""

from lifelines import CoxPHFitter
import pandas as pd
import scipy.stats
import numpy as np

def pharma_survive(i):
    
    # Importing the dataset
    dataset = pd.read_csv(r'H:/Juan Lu/Data/Coxib/427.csv')
    n_outcome = dataset.columns.get_loc("combined")

    X = dataset
    y = dataset.iloc[:,n_outcome].values
    print(sum(y))
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    
    cph = CoxPHFitter()
    cph.fit(X_train, duration_col='duration',event_col='combined',show_progress=True)
    #    a = cph.print_summary()
    cph.plot(hazard_ratios=True)
    
    pred = 1 - cph.predict_survival_function(X_test, times=[365.])
    
    from sklearn import metrics
    y_pred = pred.to_numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred.T)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    
    youden_index = np.argmax(tpr - fpr)
    threshold = thresholds[youden_index]
    print( tpr[youden_index])
    print( 1 - fpr[youden_index])
    y_decid_pred = y_pred > threshold
    y_decid_pred = y_decid_pred.T
    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_decid_pred))
    
    from sklearn.metrics import precision_score
    print(precision_score(y_test, y_decid_pred))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_decid_pred)
    
    npv = cm[0][0] / (cm[0][0] + cm[1][0])
    ppv = cm[1][1] / (cm[1][1] + cm[0][1])
 
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_decid_pred))
    
    return  [tpr[np.argmax(tpr - fpr)], 1 - fpr[np.argmax(tpr - fpr)],roc_auc, f1_score(y_test, y_decid_pred), npv, ppv ]

score = []
sensitivity = []
specificity = []
f1 = []
npv = []
ppv = []
for i in range(0,10):
    result = pharma_survive(i)
    score.append(result[2])
    sensitivity.append(result[0])
    specificity.append(result[1]) 
    f1.append(result[3])
    npv.append(result[4])
    ppv.append(result[5])
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

print(mean_confidence_interval(score))
print(mean_confidence_interval(sensitivity))
print(mean_confidence_interval(specificity))
print(mean_confidence_interval(f1))
print(mean_confidence_interval(npv))
print(mean_confidence_interval(ppv))
