# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:46:59 2016

@author: Administrator
"""

import cv2
import numpy as np

def calDark(paths):
    qdark=np.array(())
    for path in paths:
        print 'processing '+path
        img=cv2.imread(path)
        height,width = img.shape[:2] 
        Min=img.min(2)
        sumMin=sum(sum(Min))
        mul=width*height
        mul=float(mul)
        sumMin=sumMin/mul
        qdark=np.append(qdark,sumMin)
    return qdark
def dark(paths_train,paths_test):
    qdark_train=calDark(paths_train)
    np.save('E:/efficiency_AVA/data/train/19qdark_train.npy',qdark_train)
    
    qdark_test=calDark(paths_test)
    np.save('E:/efficiency_AVA/data/test/19qdark_test.npy',qdark_test)
    
    print qdark_train
    print qdark_test