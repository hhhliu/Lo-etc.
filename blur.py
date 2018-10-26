# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:23:41 2016

@author: Administrator
"""

import numpy as np 
import cv2
def getBlur(paths):
    q_blur=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path, 0)
        height,width = img.shape[:2] 
        blur = cv2.GaussianBlur(img,(3,3),0) 
        f = np.fft.fft2(blur) 
        fshift = np.fft.fftshift(f) 
        
        fimg = np.log(np.abs(fshift)) 

        colC=sum(fimg>5)
        C=sum(colC)

        Ib=height*width
        Ib=float(Ib)
        qf=C/Ib
        q_blur=np.append(q_blur,qf)
    return q_blur
    

def blur(paths_train,paths_test):
    qblur_train=getBlur(paths_train)   
    np.save('E:/efficiency_AVA/data/train/18qblur_train.npy',qblur_train)

    qblur_test=getBlur(paths_test)   
    np.save('E:/efficiency_AVA/data/test/18qblur_test.npy',qblur_test)

    print qblur_train
    print qblur_test