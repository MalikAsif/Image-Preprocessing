# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:20:01 2020

@author: Asrock
"""

import os, shutil
import cv2
import numpy as np       
def gen_path(p):
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
        nn = p+ file
        for s in [doc for doc in os.listdir(nn)]:
          filename = p +file+'/'+s 
          img = cv2.imread(filename)
          resized_image = cv2.resize(img, (224, 224)) 
          filename = p +file+'/'+s
          cv2.imwrite(filename, resized_image) 
        


path = "D:/Breast Cancer/asif/Extra-Work/Dataset/" 
gen_path(path)