# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:36:17 2018

@author: Asif
"""
import os, shutil
moveto = "D:/Breast Cancer/asif/ddsm_new/cancer/"    
             
def gen_path(p):
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p +file
          for case in [doc for doc in os.listdir(path) ]:
            
              if ".OVERLAY" in case:
                  
                  OverlayFile= case
                 
                  newfile= OverlayFile.split('.OVERLAY') 
                  PNGFile= path+'/'+newfile[0]+'.png';
                  print(PNGFile)
                  shutil.move(PNGFile,moveto)
             


pth = 'D:/Breast Cancer/asif/ddsm/cancers/'
gen_path(pth)