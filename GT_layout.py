# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 22:58:36 2016

@author: Administrator
"""

import cv2
import numpy as np

#AVA
ntrain_high=12771
ntrain_low=12771
ntest_high=12771
ntest_low=12771
ntrain = 25542
ntest =25542
count = 51084

def calDiff(matrix,height,width):
    sumVertical=0
    sumhorizontal=0
    step1=width/6
    step2=height/6
    for i in range(0,height):
        for j in range(0,step1):
            diff=matrix[i][j+5*step1]-matrix[i][j]
            sumVertical=sumVertical+diff
   
    for i in range(0,step2):
        for j in range(0,width):
            diff=matrix[i+5*step2][j]-matrix[i][j]
            sumhorizontal=sumhorizontal+diff
    sumDiff=sumVertical+sumhorizontal
    return sumDiff
    
def calGlobal(paths):
    qH=np.array([])
    qS=np.array([])
    qV=np.array([])
    qHSV=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path)
        height,width = img.shape[:2]
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H,S,V=cv2.split(HSV) 
        HSV=H+S+V
        qh=calDiff(H,height,width)
        qH=np.append(qH,qh)
        qs=calDiff(S,height,width)
        qS=np.append(qS,qs)
        qv=calDiff(V,height,width)
        qV=np.append(qV,qv)
        qhsv=calDiff(HSV,height,width)        
        qHSV=np.append(qHSV,qhsv)       
    return qH,qS,qV,qHSV


def GT_layout(paths_train,paths_test):    
    qgh_train,qgs_train,qgv_train,qghsv_train=calGlobal(paths_train)
    np.save('E:/efficiency_AVA/data/train/10qgh_train.npy',qgh_train)
    np.save('E:/efficiency_AVA/data/train/11qgs_train.npy',qgs_train)
    np.save('E:/efficiency_AVA/data/train/12qgv_train.npy',qgv_train)
    np.save('E:/efficiency_AVA/data/train/13qghsv_train.npy',qghsv_train)
    
    qgh_test,qgs_test,qgv_test,qghsv_test=calGlobal(paths_test)
    np.save('E:/efficiency_AVA/data/test/10qgh_test.npy',qgh_test)
    np.save('E:/efficiency_AVA/data/test/11qgs_test.npy',qgs_test)
    np.save('E:/efficiency_AVA/data/test/12qgv_test.npy',qgv_test)
    np.save('E:/efficiency_AVA/data/test/13qghsv_test.npy',qghsv_test)
    print 'following:'
    print qgh_train
    print qgh_test