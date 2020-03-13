# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:36:17 2018

@author: Asif
"""



import os
import subprocess
import time
import matlab.engine

def SelectROI(path, case):
     new_path = path + '\\' + case
     imageOutputFileFormat = '*.png';
     for file in [doc for doc in os.listdir(new_path)]:
           filepath= new_path+'\\'+file 
           if ".OVERLAY" in file: 
                OverlayFile= file
                newfile= file.split('.OVERLAY')
                PNGFile= newfile[0]+'.png'
                print(OverlayFile)
                [bnd_c,bnd_r] = eng.readBoundary(overlayName, 1);
              #filenames = 
             
def gen_path(p):
    file_list1 = []
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p + "\\" + file
          for case in [doc for doc in os.listdir(path) if '.' not in doc]:
              SelectROI(path,case) 
               
              
                
               
pth = 'D:\\BreastCancerThesis\\DDSM2\\'
eng = matlab.engine.start_matlab()
gen_path(pth)