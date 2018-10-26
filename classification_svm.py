# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:34:05 2016

@author: Administrator
"""
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score

#AVA
feature_vim=24
train_feature=np.load('E:/efficiency_AVA/data/trainfeature.npy')
train_label=np.load('E:/efficiency_AVA/data/trainlabel.npy')
test_feature=np.load('E:/efficiency_AVA/data/testfeature.npy')
test_label=np.load('E:/efficiency_AVA/data/testlabel.npy')


clf = svm.LinearSVC(C=2**-3)
#clf=svm.SVC()
clf.fit(train_feature,train_label)
predict_label = clf.fit(train_feature,train_label).decision_function(test_feature)
predict_label1=clf.predict(test_feature)
print predict_label
print predict_label1
np.save('C:/Users/Administrator/Desktop/dx/predict_AVA.npy',predict_label)
fpr,tpr,thresholds  = roc_curve(test_label,predict_label)
roc_auc = auc(fpr,tpr)

plt.figure()
lw=2 #line width
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

score = clf.score(test_feature,test_label)
print 'Accuracy: %0.4f' % score