# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:34:52 2016

@author: Administrator
"""

import random
def spilDataset(dataset):
    spilRatio=0.5
    trainSize=int(len(dataset)*spilRatio) 
    trainSet=[]  
    testSet=list(dataset)
    while len(trainSet) < trainSize:
        index=random.randrange(len(testSet))
        trainSet.append(testSet.pop(index))
    return [trainSet,testSet]
    