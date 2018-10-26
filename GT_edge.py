# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:14:18 2016

@author: Administrator
"""

import cv2
import numpy as np

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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv) 
        gray_lapH = cv2.Laplacian(h,cv2.CV_16S,ksize = 3)  
        dstH = cv2.convertScaleAbs(gray_lapH)          
        gray_lapS = cv2.Laplacian(s,cv2.CV_16S,ksize = 3)  
        dstS = cv2.convertScaleAbs(gray_lapS)        
        gray_lapV = cv2.Laplacian(v,cv2.CV_16S,ksize = 3)  
        dstV = cv2.convertScaleAbs(gray_lapV)   
        dstHSV=dstH+dstS+dstV
        
        qh=calDiff(dstH,height,width)
        qH=np.append(qH,qh)
        qs=calDiff(dstS,height,width)
        qS=np.append(qS,qs)
        qv=calDiff(dstV,height,width)
        qV=np.append(qV,qv)
        qhsv=calDiff(dstHSV,height,width)        
        qHSV=np.append(qHSV,qhsv)       
    return qH,qS,qV,qHSV


def GT_edge(paths_train,paths_test):
    qgh_train,qgs_train,qgv_train,qghsv_train=calGlobal(paths_train)
    np.save('E:/efficiency_AVA/data/train/14qgeh_train.npy',qgh_train)
    np.save('E:/efficiency_AVA/data/train/15qges_train.npy',qgs_train)
    np.save('E:/efficiency_AVA/data/train/16qgev_train.npy',qgv_train)
    np.save('E:/efficiency_AVA/data/train/17qgehsv_train.npy',qghsv_train)
    
    qgh_test,qgs_test,qgv_test,qghsv_test=calGlobal(paths_test)
    np.save('E:/efficiency_AVA/data/test/14qgeh_test.npy',qgh_test)
    np.save('E:/efficiency_AVA/data/test/15qges_test.npy',qgs_test)
    np.save('E:/efficiency_AVA/data/test/16qgev_test.npy',qgv_test)
    np.save('E:/efficiency_AVA/data/test/17qgehsv_test.npy',qghsv_test)
    
    print qgh_train
    print qgh_test