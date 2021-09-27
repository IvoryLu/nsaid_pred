# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:06:32 2018

@author: 00098223
"""

#Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def svm(i):
    
    dataset = pd.read_csv('H:/Juan Lu/Data/Coxib/422.csv')
    print(dataset.columns)
    # Death / ACS
    n_outcome = dataset.columns.get_loc("combined")
    X = dataset.iloc[:,0:n_outcome].values
    y = dataset.iloc[:,n_outcome].values
    print(sum(y))
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
     
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import GridSearchCV
    
    from sklearn.svm import SVC
    #from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    
    classifier = SVC(
                     C =3,# 3 for combined
                     kernel = 'rbf',
#                     kernel = 'poly',
                     random_state = 42,
                     #multi_class = 'crammer_singer',
                     gamma='scale',
#                     gamma=100,
                     class_weight = 'balanced',
                     # max_iter = 2000
                     )
                     #probability=True)
    
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.decision_function(X_test)

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    sens_spec = 0
    for i in range(2):
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    youden_index = np.argmax(tpr[1] - fpr[1])
    threshold = thresholds[youden_index]
    y_decid_pred = y_pred > threshold

    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_decid_pred))
    
    from sklearn.metrics import precision_score
    print(precision_score(y_test, y_decid_pred))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_decid_pred)
    
    npv = cm[0][0] / (cm[0][0] + cm[1][0])
    ppv = cm[1][1] / (cm[1][1] + cm[0][1])
    print( tpr[1][np.argmax(tpr[1] - fpr[1])])
    print( 1 - fpr[1][np.argmax(tpr[1] - fpr[1])])
    print(roc_auc_score(y_test, y_pred))

    return  [tpr[1][np.argmax(tpr[1] - fpr[1])], 1 - fpr[1][np.argmax(tpr[1] - fpr[1])],roc_auc_score(y_test, y_pred), npv, ppv, f1_score(y_test, y_decid_pred)]


score = []
sensitivity = []
specificity = []
f1 = []
npv = []
ppv = []
for i in range(0, 50):

    result = svm(i)
    score.append(result[2])
    sensitivity.append(result[0])
    specificity.append(result[1])    
    f1.append(result[3])
    npv.append(result[4])
    ppv.append(result[5])
    
print(mean_confidence_interval(score))
print(mean_confidence_interval(sensitivity))
print(mean_confidence_interval(specificity))
print(mean_confidence_interval(f1))
print(mean_confidence_interval(npv))
print(mean_confidence_interval(ppv))

df = pd.DataFrame(data = score, columns=["AUC"])

df['sensitivity'] = sensitivity
df['specificity'] = specificity
df['f1'] = f1
df['npv'] = npv
df['ppv'] = ppv
file_name = 'H:/Juan Lu/Data/Coxib/422_acs_svm_' + str(i) + '.csv'
df.to_csv(file_name, index=False)