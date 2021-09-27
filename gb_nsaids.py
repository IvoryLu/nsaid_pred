# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:51:58 2020

@author: 00098223
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def gb(i, n_estimator=450):
    # # # # Importing the dataset
    dataset = pd.read_csv(r'H:\Juan Lu\Data\Coxib\422.csv')
    
    n_outcome = dataset.columns.get_loc("combined")
    X = dataset.iloc[:,0:n_outcome].values
    y = dataset.iloc[:,n_outcome].values
    print(sum(y))
    print(i)
    
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
        
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    from sklearn.ensemble import GradientBoostingClassifier
     
    classifier = GradientBoostingClassifier(n_estimators=n_estimator, learning_rate=0.01, 
                                            max_depth=10,
                                            min_samples_leaf = 700,
                                            min_samples_split = 50,
                                            subsample = 0.8,
                                            max_features='sqrt',
                                            warm_start= True,
                                            random_state=0).fit(X_train, y_train)

    
    
    y_pred = classifier.decision_function(X_test)
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(2):
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    youden_index = np.argmax(tpr[1] - fpr[1])
    threshold = thresholds[youden_index]
    print( tpr[1][youden_index])
    print( 1 - fpr[1][youden_index])
    print(roc_auc_score(y_test, y_pred))
    y_decid_pred = y_pred > threshold
    
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
    
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    
    from sklearn.metrics import plot_precision_recall_curve
    disp = plot_precision_recall_curve(classifier, X_test, y_test)

    
    # Feature Importance
    feature_importance = classifier.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    nsaids = ['Ketoprofen', 'Ibuprofen', 'Piroxicam', 'Naproxen', 'Indometacin', 'Diclofenac', 'Rofecoxib',
              'Meloxicam', 'Celecoxib','Sulindac','Piroxicam']
    sorted_idx = [i for i in sorted_idx if np.array(dataset.columns)[i] in nsaids]
    sorted_idx = np.array(sorted_idx)
    # sorted_idx = sorted_idx[10:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(dataset.columns)[sorted_idx],fontsize=8)
    plt.title('GBM Feature Importance (All-cause death)')
    plt.figure()
    
    # ROC CURVE
    # lw = 1
    # plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic'+dataset.columns[-1])
    # plt.legend(loc="lower right")
    # plt.show()

    probability = np.round(y_pred,3)

    df = pd.DataFrame(data = probability, columns=["proability"] )

    # sns.displot(df, x = "proability", hue=y_test, kind="kde", fill=True, palette="husl")
    
    # return roc_auc_score(y_test, y_pred)
    return  [tpr[1][np.argmax(tpr[1] - fpr[1])], 1 - fpr[1][np.argmax(tpr[1] - fpr[1])],roc_auc_score(y_test, y_pred), f1_score(y_test, y_decid_pred), npv, ppv ]

   
score = []
sensitivity = []
specificity = []
f1 = []
npv = []
ppv = []
for i in range(0,50):
    # n_estimators = 400
    # score.append(gb(i))
    result = gb(i)
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

#%%
df = pd.DataFrame(data = score, columns=["AUC"] )

df['sensitivity'] = sensitivity
df['specificity'] = specificity

file_name = 'H:/Juan Lu/Data/Coxib/result_gbm_' + str(i) + '.csv'
df.to_csv(file_name, index=False)