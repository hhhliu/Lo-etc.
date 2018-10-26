# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:45:53 2016

@author: Administrator
"""

import getPath
#import colorPalette
import layoutComposition
import edgeComposition
import GT_layout
import GT_edge
import blur
import dark
import Contrasts
import HSVcounts


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

layoutComposition.layout(paths_trainhigh,paths_testhigh,paths_trainlow,paths_testlow,paths_train,paths_test)
edgeComposition.EC(paths_trainhigh,paths_testhigh,paths_trainlow,paths_testlow,paths_train,paths_test)

GT_layout.GT_layout(paths_train,paths_test)
GT_edge.GT_edge(paths_train,paths_test)
blur.blur(paths_train,paths_test)
dark.dark(paths_train,paths_test)
Contrasts.contrast(paths_train,paths_test)
HSVcounts.hsvcounts(paths_train,paths_test)

#colorPalette.colorPalette(paths_train,paths_test)