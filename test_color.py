# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:33:48 2016

@author: Administrator
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

#animal
ntrain_high=476
ntrain_low=1146
ntest_high=476
ntest_low=1146
ntrain = 1622
ntest = 1622
count = 3244
    
def classify(counts,dataSet_train,dataSet,labels):
    print 'classify'
    q_color=np.array([])
    k=5
    for i in range(counts):
        testsample=dataSet[i]
        testsample=np.array(testsample).reshape((1,-1))
        neigh=NearestNeighbors(n_neighbors=k)
        neigh.fit(dataSet_train)
        distances,index= neigh.kneighbors(testsample)
        npr=0
        nsn=0
        for j in range(k):            
            vote = labels[index[0][j]]
            if vote==0:
                nsn=nsn+1
            if vote==1:
                npr=npr+1
        qcd=npr-nsn
        q_color=np.append(q_color,qcd)
    return q_color
    
def colorPalette(paths_train,paths_test):  
    #data=createC()
    labels=[] 
    for j in range(ntrain_high):  
        labels.append(1)
    for k in range(ntrain_low):
        labels.append(0)
    
    featureData_train=np.load('E:/efficiency/data/animal/featureData_train.npy')
    featureData_test=np.load('E:/efficiency/data/animal/featureData_test.npy')
    feature_train=np.reshape(featureData_train,(ntrain,15))
    feature_test=np.reshape(featureData_test,(ntest,15))
    print feature_train.shape
    print feature_test.shape
    
    qcolor_train=classify(ntrain,feature_train,feature_train,labels)  
    np.save('E:/efficiency/data/animal/train/1qcolor_train.npy',qcolor_train)
    qcolor_test=classify(ntest,feature_train,feature_test,labels)
    np.save('E:/efficiency/data/animal/test/1qcolor_test.npy',qcolor_test)
    print qcolor_train
    print qcolor_test
    
paths_train=np.load('E:/efficiency/data/animal/paths/paths_train.npy')
paths_test=np.load('E:/efficiency/data/animal/paths/paths_test.npy')
colorPalette(paths_train,paths_test)