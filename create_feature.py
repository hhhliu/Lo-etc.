# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:14:10 2017

@author: Administrator
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,accuracy_score
import getPath

ntrain_high=12771
ntrain_low=12771
ntest_high=12771
ntest_low=12771
ntrain = 25542
ntest = 25542
count = 51084
feature_dim=24

feature_train=np.zeros((feature_dim,ntrain))
root_train='E:/efficiency_AVA/data/train'
paths_train,count_train=getPath.getPath(root_train)
i=0
for path in paths_train:
    feature=np.load(path) 
    feature=np.array(feature)
    feature_train[i]=feature
    i=i+1
train_feature=np.transpose(feature_train)   
np.save('E:/efficiency_AVA/data/trainfeature.npy',train_feature)

feature_test = np.zeros((feature_dim,ntest))
root_test='E:/efficiency_AVA/data/test'
paths_test,count_test=getPath.getPath(root_test)
i=0
for path in paths_test:
    feature=np.load(path)  
    feature=np.array(feature)
    feature_test[i]=feature
    i=i+1
test_feature=np.transpose(feature_test)  
np.save('E:/efficiency_AVA/data/testfeature.npy',test_feature)

label_train=np.array([])
for i in range(ntrain_high):
    label_train=np.append(label_train,1) 
for j in range(ntrain_low):
    label_train=np.append(label_train,0)
train_label=np.transpose(label_train)
np.save('E:/efficiency_AVA/data/trainlabel.npy',train_label)

label_test=np.array([])
for i in range(ntest_high):
    label_test=np.append(label_test,1)  
for j in range(ntest_low):
    label_test=np.append(label_test,0)
test_label=np.transpose(label_test)
np.save('E:/efficiency_AVA/data/testlabel.npy',test_label)