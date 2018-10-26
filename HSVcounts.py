# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 09:20:08 2016

@author: Administrator
"""

import cv2
import numpy as np

def calHist(img):
    height,width = img.shape[:2] 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(hsv)
    histH=cv2.calcHist([img],[0],None,[16],[0.0,180.0])
    histS=cv2.calcHist([img],[1],None,[16],[0.0,256.0])
    histV=cv2.calcHist([img],[2],None,[16],[0.0,256.0])
    return histH,histS,histV 
    
def calHSVcounts(paths):
    qH=np.array([])
    qS=np.array([])
    qV=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path)
        histH,histS,histV=calHist(img)
        qhsvH=sum(histH!=0)
        qhsvS=sum(histS!=0)
        qhsvV=sum(histV!=0)
        qH=np.append(qH,qhsvH)
        qS=np.append(qS,qhsvS)
        qV=np.append(qV,qhsvV)
    return qH,qS,qV
  
  
def hsvcounts(paths_train,paths_test):
    qh_train,qs_train,qv_train=calHSVcounts(paths_train)
    qh_test,qs_test,qv_test=calHSVcounts(paths_test)

    np.save('E:/efficiency_AVA/data/train/22qhcounts_train.npy',qh_train)
    np.save('E:/efficiency_AVA/data/test/22qhcounts_test.npy',qh_test)
    np.save('E:/efficiency_AVA/data/train/23qscounts_train.npy',qs_train)
    np.save('E:/efficiency_AVA/data/test/23qscounts_test.npy',qs_test)
    np.save('E:/efficiency_AVA/data/train/24qvcounts_train.npy',qv_train)
    np.save('E:/efficiency_AVA/data/test/24qvcounts_test.npy',qv_test)

    print qh_train
    print qh_test
    print qs_train
    print qs_test
    print qv_train
    print qv_test