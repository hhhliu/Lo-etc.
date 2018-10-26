# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:54:37 2016

@author: Administrator
"""

import os
import os.path
import numpy as np
def getPath(rootdir):
    path=np.array(())
    count=0
    for root,dirs,files in os.walk(rootdir):
        for fn in files:
            f= root+'/'+fn
            path=np.append(path,f)
            count=count+1
    return path,count