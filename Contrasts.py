# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:18:12 2016

@author: Administrator
"""
import cv2
import numpy as np
def getRGBcontrast(paths):
    q_contrast=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path)
        height, width = img.shape[:2] 
        histb=cv2.calcHist([img],[0],None,[256],[0.0,256.0])
        histg=cv2.calcHist([img],[1],None,[256],[0.0,256.0])
        histr=cv2.calcHist([img],[2],None,[256],[0.0,256.0])
        hist=histb+histg+histr
        flag= 0.01*3*height*width
        sum1=0
        sum2=0
        for i in range(256):
            sum1=sum1+hist[i]
            if sum1>=flag:
                left=i
                break
        for j in range(256):
            sum2=sum2+hist[256-j-1]
            if sum2>=flag:
                right=256-j-1
                break
        qct=right-left
        q_contrast=np.append(q_contrast,qct)
    return q_contrast
    
def getGraycontrast(paths):
    q_contrast=np.array([])
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path,0)
        height, width = img.shape[:2] 
        hist=cv2.calcHist([img],[0],None,[256],[0.0,256.0])
        flag= 0.01*height*width
        sum1=0
        sum2=0
        for i in range(256):
            sum1=sum1+hist[i]
            if sum1>=flag:
                left=i
                break
        for j in range(256):
            sum2=sum2+hist[256-j-1]
            if sum2>=flag:
                right=256-j-1
                break
        qct=right-left
        q_contrast=np.append(q_contrast,qct)
    return q_contrast
    
def contrast(paths_train,paths_test):
    qcontrastRGB_train=getRGBcontrast(paths_train)
    np.save('E:/efficiency_AVA/data/train/20qcontrastRGB_train.npy',qcontrastRGB_train)
    
    qcontrastRGB_test=getRGBcontrast(paths_test)
    np.save('E:/efficiency_AVA/data/test/20qcontrastRGB_test.npy',qcontrastRGB_test)

    qcontrastGray_train=getGraycontrast(paths_train)
    np.save('E:/efficiency_AVA/data/train/21qcontrastGray_train.npy',qcontrastGray_train)
    
    qcontrastGray_test=getGraycontrast(paths_test)
    np.save('E:/efficiency_AVA/data/test/21qcontrastGray_test.npy',qcontrastGray_test)
    
    print qcontrastRGB_train
    print qcontrastRGB_test
    print qcontrastGray_train
    print qcontrastGray_test