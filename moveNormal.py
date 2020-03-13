# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:36:17 2018

@author: Asif
"""
import os, shutil
moveto = "D:/Breast Cancer/asif/ddsm_new/normal/"    
             
def gen_path(p):
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p +file
          for case in [doc for doc in os.listdir(path) ]:
              src = path+ '/' + case
              print(src)
              shutil.move(src,moveto)
             
             


pth = 'D:/Breast Cancer/asif/ddsm/normals/'
gen_path(pth)