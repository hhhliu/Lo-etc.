# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:46:15 2016

@author: Administrator
"""

import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import scipy.spatial
import getPath

#AVA
ntrain_high=12771
ntrain_low=12771
ntest_high=12771
ntest_low=12771
ntrain = 25542
ntest =25542
count = 51084

#histogram
def extractfeature(img):
    print 'extracting'
    height, width = img.shape[:2] 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    H,S,V=cv2.split(hsv)
    Hist=np.zeros((1,4096))   
    for x in range(height):
        for y in range(width):
            h=H[x][y]
            s=S[x][y]
            v=V[x][y]
            h=h/16
            s=s/16
            v=v/16
            index=h*16*16+s*16+v
            Hist[0][index]=Hist[0][index]+1
    return Hist      
        
def createC():
    C=np.zeros((4096,3))
    m=0  
    for x in range(8,255,16):        
        for y in range(8,255,16):            
            for z in range(8,255,16):                
                C[m][0]=x
                C[m][1]=y
                C[m][2]=z
                m=m+1  
    return C

def cluster_centroids(hist,data,clusters):
    results=[]
    for i in range(5): 
        if sum(hist[0][clusters==i]==0):
            results.append(np.average(data[clusters==i],axis=0))
        else:
            results.append(np.average(data[clusters==i],axis=0,weights=np.asarray(hist[0][clusters==i])))
    return results
    
def weighted_kmeans(hist,data, k=None, centroids=None, steps=20):
    centroids = data[np.random.choice(np.arange(len(data)), k, False)]
    for i in range(max(steps, 1)):
        sqdists = scipy.spatial.distance.cdist(centroids, data,metric='euclidean')
        clusters = np.argmin(sqdists, axis=0)
        new_centroids = cluster_centroids(hist,data, clusters)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def findDocolor(data,paths,counts):
    colorFeature=np.array([])
    for path in paths:
        print path
        img=cv2.imread(path)
        hist=extractfeature(img)
        clusters,centroids=weighted_kmeans(hist,data,k=5)
        feature=np.array([])
        for i in range(5):
            centroids_array=np.tile(centroids[i],(data[clusters==i].shape[0],1))
            diffMat=centroids_array-data[clusters==i]
            absDiffMat =abs(diffMat) 
            distances =1.0/absDiffMat.sum(axis=1)   
            Dom=0.5*hist[0][clusters==i]+0.5*distances
            index = np.argmax(Dom, axis=0)
            color=data[index]
            for j in range(3):
                feature=np.append(feature,color[j])
        print feature
        colorFeature=np.append(colorFeature,feature)
    return colorFeature
    
def classify(counts,dataSet_train,dataSet,labels):
    print 'classify'
    q_color=np.array([])
    k=5
    for i in range(counts):
        print i
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
    """featureData_train=findDocolor(data,paths_train,ntrain)
    np.save('E:/efficiency_AVA/data/featureData_train.npy',featureData_train)
    featureData_test=findDocolor(data,paths_test,ntest)
    np.save('E:/efficiency_AVA/data/featureData_test.npy',featureData_test)"""  
    
    featureData_train=np.load('E:/efficiency_AVA/data/featureData_train.npy')    
    featureData_test=np.load('E:/efficiency_AVA/data/featureData_test.npy')
    featureData_train=np.reshape(featureData_train,(ntrain,15))
    featureData_test=np.reshape(featureData_test,(ntest,15))
    
    qcolor_train=classify(ntrain,featureData_train,featureData_train,labels)  
    np.save('E:/efficiency_AVA/data/train/1qcolor_train.npy',qcolor_train)
    qcolor_test=classify(ntest,featureData_train,featureData_test,labels)
    np.save('E:/efficiency_AVA/data/test/1qcolor_test.npy',qcolor_test)
    print qcolor_train
    print qcolor_test
    
root_train = 'E:/ImageDataset_AVA/train/'  
root_test = 'E:/ImageDataset_AVA/test/'  
paths_train,counts_train=getPath.getPath(root_train)
paths_test,counts_test=getPath.getPath(root_test)
colorPalette(paths_train,paths_test)