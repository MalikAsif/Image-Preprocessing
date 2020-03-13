
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:36:17 2018

@author: Asif
"""

import os, shutil
import cv2
import numpy as np       
def gen_path(p):
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
        nn = p+ file
       
        for s in [doc for doc in os.listdir(nn)]:
          d = p +file+'/'+s
          img = cv2.imread(d)
          avg = np.average(img)
          if avg < 30:
             os.remove(d)
         


path = "D:/Breast Cancer/asif/ddsm_new/" 
gen_path(path)