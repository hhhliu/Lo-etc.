# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:53:22 2016

@author: Administrator
"""

import numpy as np
import cv2 
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
    sumResized=sum(sum(resized))
    sumResized=float(sumResized)
    resized=resized/sumResized
    return resized

def train(paths,counts):
    sumH=np.zeros((100,100))
    sumS=np.zeros((100,100))
    sumV=np.zeros((100,100))
    sumHSV=np.zeros((100,100))
    counts=float(counts)
    for path in paths:
        print 'training '+path
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        height, width = img.shape[:2] 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        
        h,s,v=cv2.split(hsv)  
        hsv=h+s+v

        if sum(sum(h))==0:
            H=np.zeros((100,100))
        else:
            H=normalization(h)
            
        if sum(sum(s))==0:
            S=np.zeros((100,100))
        else:
            S=normalization(s)
            
        if np.all(np.isnan(S)):
            S=np.zeros((100,100))
   
            
        if sum(sum(v))==0:
            V=np.zeros((100,100))
        else:
            V=normalization(v) 
            
        if sum(sum(hsv))==0:
            HSV=np.zeros((100,100)) 
        else:
            HSV=normalization(hsv)         
            
        sumH=sumH+H
        sumS=sumS+S
        sumV=sumV+V
        sumHSV=sumHSV+HSV
        
    avgH=sumH/counts
    avgS=sumS/counts
    avgV=sumV/counts
    avgHSV=sumHSV/counts
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
        hsv=h+s+v

        if sum(sum(h))==0:
            H=np.zeros((100,100))
        else:
            H=normalization(h)
            
        if sum(sum(s))==0:
            S=np.zeros((100,100))
        else:
            S=normalization(s)
        if np.all(np.isnan(S)):
            S=np.zeros((100,100))
            
        if sum(sum(v))==0:
            V=np.zeros((100,100))
        else:
            V=normalization(v) 
            
        if sum(sum(hsv))==0:
            HSV=np.zeros((100,100)) 
        else:
            HSV=normalization(hsv)    
            
        
        dhh=sum(sum(abs(H-avgh_trainhigh)))
        dhl=sum(sum(abs(H-avgh_trainlow)))
        qh=dhl-dhh
        qH= np.append(qH,qh)
        
        dsh=sum(sum(abs(S-avgs_trainhigh)))
        dsl=sum(sum(abs(S-avgs_trainlow)))
        qs=dsl-dsh
        qS=np.append(qS,qs)
        if np.all(np.isnan(qS)):
            print path
            break
            return
        dvh=sum(sum(abs(V-avgv_trainhigh)))
        dvl=sum(sum(abs(V-avgv_trainlow)))
        qv=dvl-dvh
        qV=np.append(qV,qv)
        
        dhsvh=sum(sum(abs(HSV-avghsv_trainhigh)))
        dhsvl=sum(sum(abs(HSV-avghsv_trainlow)))
        qhsv=dhsvl-dhsvh
        qHSV=np.append(qHSV,qhsv)
        
    return qH,qS,qV,qHSV
 
def layout(paths_trainhigh,paths_testhigh,paths_trainlow,paths_testlow,paths_train,paths_test):
    
    avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh=train(paths_trainhigh,ntrain_high)      
    avgh_testhigh,avgs_testhigh,avgv_testhigh,avghsv_testhigh=train(paths_testhigh,ntest_high)

    avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow=train(paths_trainlow,ntrain_low)
    avgh_testlow,avgs_testlow,avgv_testlow,avghsv_testlow=train(paths_testlow,ntest_low)
    

    qH_train,qS_train,qV_train,qHSV_train=calLayout(paths_train,avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh,avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow)
    np.save('E:/efficiency_AVA/data/train/2qH',qH_train)
    np.save('E:/efficiency_AVA/data/train/3qS',qS_train)
    np.save('E:/efficiency_AVA/data/train/4qV',qV_train)
    np.save('E:/efficiency_AVA/data/train/5qHSV',qHSV_train)

    qH_test,qS_test,qV_test,qHSV_test=calLayout(paths_test,avgh_trainhigh,avgs_trainhigh,avgv_trainhigh,avghsv_trainhigh,avgh_trainlow,avgs_trainlow,avgv_trainlow,avghsv_trainlow)
    np.save('E:/efficiency_AVA/data/test/2qH',qH_test)
    np.save('E:/efficiency_AVA/data/test/3qS',qS_test)
    np.save('E:/efficiency_AVA/data/test/4qV',qV_test)
    np.save('E:/efficiency_AVA/data/test/5qHSV',qHSV_test)
    
    
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
layout(paths_trainhigh,paths_testhigh,paths_trainlow,paths_testlow,paths_train,paths_test)