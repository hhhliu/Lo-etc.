# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 22:02:28 2016

@author: Administrator
"""

import cv2
import numpy as np
import getPath

#AVA
ntrain_high=12771
ntrain_low=12771
ntest_high=12771
ntest_low=12771
ntrain = 25542
ntest =25542
count = 51084

def normalization(matrix):
    resized = cv2.resize(matrix, (100,100), interpolation=cv2.INTER_AREA)
    sumResized=sum(map(sum,resized))
    sumResized=float(sumResized)
    resized=resized/sumResized
    return resized

def avg(matrix,counts):
    counts=float(counts)
    averaged=matrix/counts
    return averaged

def train(paths,counts):
    sumH=np.zeros((100,100))
    sumS=np.zeros((100,100))
    sumV=np.zeros((100,100))
    sumHSV=np.zeros((100,100))
    for path in paths:
        print 'training: '+path
        img=cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)  
        gray_lapH = cv2.Laplacian(h,cv2.CV_16S,ksize = 3)  
        dstH = cv2.convertScaleAbs(gray_lapH)          
        gray_lapS = cv2.Laplacian(s,cv2.CV_16S,ksize = 3)  
        dstS = cv2.convertScaleAbs(gray_lapS)        
        gray_lapV = cv2.Laplacian(v,cv2.CV_16S,ksize = 3)  
        dstV = cv2.convertScaleAbs(gray_lapV)   
        dstHSV=dstH+dstS+dstV
        
        a=sum(sum(dstH))
        b=sum(sum(dstS))
        c=sum(sum(dstV))
        H=normalization(dstH)
        S=normalization(dstS)
        V=normalization(dstV)        
        HSV=normalization(dstHSV)

        if a==0 & b==0 & c==0:
            H=np.zeros((100,100))
            S=np.zeros((100,100))
            V=np.zeros((100,100))
            HSV=np.zeros((100,100))
            
        if a==0:
            H=np.zeros((100,100))
        if b==0:
            S=np.zeros((100,100))            
        if c==0:           
            V=np.zeros((100,100))
        
        sumH=sumH+H
        sumS=sumS+S
        sumV=sumV+V
        sumHSV=sumHSV+HSV
    avgH=avg(sumH,counts)
    avgS=avg(sumS,counts)
    avgV=avg(sumV,counts)
    avgHSV=avg(sumHSV,counts)
    return avgH,avgS,avgV,avgHSV
        
def calLayout(paths,avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh,avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow):
    qH=np.array(())
    qS=np.array(())
    qV=np.array(())
    qHSV=np.array(())
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)  
        gray_lapH = cv2.Laplacian(h,cv2.CV_16S,ksize = 3)  
        dstH = cv2.convertScaleAbs(gray_lapH)          
        gray_lapS = cv2.Laplacian(s,cv2.CV_16S,ksize = 3)  
        dstS = cv2.convertScaleAbs(gray_lapS)        
        gray_lapV = cv2.Laplacian(v,cv2.CV_16S,ksize = 3)  
        dstV = cv2.convertScaleAbs(gray_lapV)   
        dstHSV=dstH+dstS+dstV
        
        a=sum(sum(dstH))
        b=sum(sum(dstS))
        c=sum(sum(dstV))
        H=normalization(dstH)
        S=normalization(dstS)
        V=normalization(dstV)        
        HSV=normalization(dstHSV)

        if a==0 & b==0 & c==0:
            H=np.zeros((100,100))
            S=np.zeros((100,100))
            V=np.zeros((100,100))
            HSV=np.zeros((100,100))
            
        if a==0:
            H=np.zeros((100,100))
        if b==0:
            S=np.zeros((100,100))            
        if c==0:           
            V=np.zeros((100,100))    
        
        dhh=sum(sum(abs(H-avgh_trainhigh)))
        dhl=sum(sum(abs(H-avgh_trainlow)))
        qh=dhl-dhh
        qH= np.append(qH,qh)
        
        dsh=sum(sum(abs(S-avgs_trainhigh)))
        dsl=sum(sum(abs(S-avgs_trainlow)))
        qs=dsl-dsh
        qS=np.append(qS,qs)
        
        dvh=sum(sum(abs(V-avgv_trainhigh)))
        dvl=sum(sum(abs(V-avgv_trainlow)))
        qv=dvl-dvh
        qV=np.append(qV,qv)
        
        dhsvh=sum(sum(abs(HSV-avghsv_trainhigh)))
        dhsvl=sum(sum(abs(HSV-avghsv_trainlow)))
        qhsv=dhsvl-dhsvh
        qHSV=np.append(qHSV,qhsv)
        
    return qH,qS,qV,qHSV
    
def EC(paths_trainhigh,paths_testhigh,paths_trainlow,paths_testlow,paths_train,paths_test):
    
    avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh=train(paths_trainhigh,ntrain_high)      
    avgh_testhigh,avgs_testhigh,avgv_testhigh,avghsv_testhigh=train(paths_testhigh,ntest_high)

    avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow=train(paths_trainlow,ntrain_low)
    avgh_testlow,avgs_testlow,avgv_testlow,avghsv_testlow=train(paths_testlow,ntest_low)

    qH_train,qS_train,qV_train,qHSV_train=calLayout(paths_train,avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh,avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow)
    np.save('E:/efficiency_AVA/data/train/6qH',qH_train)
    np.save('E:/efficiency_AVA/data/train/7qS',qS_train)
    np.save('E:/efficiency_AVA/data/train/8qV',qV_train)
    np.save('E:/efficiency_AVA/data/train/9qHSV',qHSV_train)

    qH_test,qS_test,qV_test,qHSV_test=calLayout(paths_test,avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh,avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow)
    np.save('E:/efficiency_AVA/data/test/6qH',qH_test)
    np.save('E:/efficiency_AVA/data/test/7qS',qS_test)
    np.save('E:/efficiency_AVA/data/test/8qV',qV_test)
    np.save('E:/efficiency_AVA/data/test/9qHSV',qHSV_test)
    
    print 'following:'
    print qH_train
    print qS_train
    print qV_train
    print qHSV_train
    
    print qH_test
    print qS_test
    print qV_test
    print qHSV_test

root_train = 'E:/ImageDataset_AVA/train/'  
root_test = 'E:/ImageDataset_AVA/test/'  
paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)

root_trainhigh='E:/ImageDataset_AVA/train/train_high'  
root_trainlow='E:/ImageDataset_AVA/train/train_low'  
root_testhigh='E:/ImageDataset_AVA/test/test_high'  
root_testlow='E:/ImageDataset_AVA/test/test_low'  

paths_trainhigh,counts_trainhigh=getPath.getPath(root_trainhigh)
paths_trainlow,counts_trainlow=getPath.getPath(root_trainlow)
paths_testhigh,counts_testhigh=getPath.getPath(root_testhigh)
paths_testlow,counts_testlow=getPath.getPath(root_testlow)
EC(paths_trainhigh,paths_testhigh,paths_trainlow,paths_testlow,paths_train,paths_test)