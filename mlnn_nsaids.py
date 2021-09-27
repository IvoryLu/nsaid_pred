# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:37:38 2019

@author: 00098223
"""

#Part 1 - Data Preprocessing

##Importing the libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect import signature
import seaborn as sns

def ann(i,maximum = 0):
    
    # Importing the dataset

    dataset = pd.read_csv(r'H:\Juan Lu\Data\Coxib\422.csv')
    n_outcome = dataset.columns.get_loc("combined")
    X = dataset.iloc[:,0:n_outcome].values
    y = dataset.iloc[:,n_outcome].values
    print(sum(y))
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
        
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
                            
    
    #Part 2 - ANN
    #Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
#    
    #Initialising the ANN
    classifier = Sequential()
        
    classifier.add(Dense(input_dim = 45,units= 2250, kernel_initializer = "uniform", activation= "relu"))
    
    classifier.add(Dense(units = 825, kernel_initializer = "uniform", activation= "relu"))
        
    classifier.add(Dense(units = 18, kernel_initializer = "uniform", activation= "relu"))#17
    
    #Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation= "sigmoid"))

    
    from keras import optimizers
    from keras.models import model_from_yaml
    opt = optimizers.Adam(lr= 0.001, beta_1=0.9, beta_2=0.999, epsilon=None,decay=0.0,amsgrad=False)
    
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
        
    from sklearn.utils import class_weight
    
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)
    
    history = classifier.fit(X_train, y_train, batch_size=6000, epochs=10, validation_split=0.1, class_weight = class_weight)
    
            
    y_pred = classifier.predict(X_test)
    # y_pred = loaded_model.predict(X_test)
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
    print(roc_auc_score(y_test, y_pred))
          
    probability = np.round(y_pred,3)
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

    df = pd.DataFrame(data = probability, columns=["proability"] )
    
    # sns.displot(df, x = "proability", hue=y_test, kind="kde", fill=True, palette="husl")
    
    # return roc_auc_score(y_test, y_pred)
    return  [tpr[1][np.argmax(tpr[1] - fpr[1])], 1 - fpr[1][np.argmax(tpr[1] - fpr[1])],roc_auc_score(y_test, y_pred), npv, ppv, f1_score(y_test, y_decid_pred)]

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#%%
score = []
sens = []
spec = []
f1 = []
npv = []
ppv = []

for i in range(0,50):  
    result = ann(i,0)
    score.append(result[2])
    sens.append(result[0])
    spec.append(result[1])
    f1.append(result[3])
    npv.append(result[4])
    ppv.append(result[5])
    
print(mean_confidence_interval(score))
print(mean_confidence_interval(sens))
print(mean_confidence_interval(spec))
print(mean_confidence_interval(f1))
print(mean_confidence_interval(npv))
print(mean_confidence_interval(ppv))