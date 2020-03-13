# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 20:05:52 2018

@author: Asif
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:29:55 2018

@author: Asif
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:36:17 2018

@author: Asif
"""
import os
import subprocess
import time
import matlab.engine
import cv2
import numpy as np
import matplotlib.pylab as plt
import scipy 
from skimage import draw
from PIL import Image, ImageDraw

#from ..._shared import warn

def SelectROI(path, case):
     new_path = path + '/' + case
     number=1
     for file in [doc for doc in os.listdir(new_path)]:
           filepath= new_path+'/'+file 
           if ".OVERLAY" in file: 
                OverlayFile= file
                newfile= file.split('.OVERLAY') 
                PNGFile= new_path+'/'+newfile[0]+'.png'
                print(PNGFile)
                print(OverlayFile) 
                bnd_c,bnd_r = eng.readBoundary(filepath, number, nargout=2);
                points= [[item[0][0], item[1][0]] for item in list(zip(np.asarray(bnd_c).T, np.asarray(bnd_r).T))]
                pts= np.array(points)
                image = cv2.imread(PNGFile);
                x_min= int(min(min(bnd_c)))
                x_max =int( max(max(bnd_c)))
                y_min= int(min(min(bnd_r)))
                y_max = int(max(max(bnd_r)))
                print(x_min)
                print(x_max)
                print(y_min)
                print(y_max)
                x_diff= x_max-x_min;
                y_diff= y_max-y_min
                roi= image[y_min:y_max,x_min:x_max]
                #img[0:x_diff,0:y_diff]= roi
                cv2.imwrite(PNGFile,roi)
                print("image saved")
             
                
               
              
             
def gen_path(p):
    file_list1 = []
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p + file
          for case in [doc for doc in os.listdir(path) if '.' not in doc]:
              SelectROI(path,case) 


pth = 'D:/BreastCancerThesis/DDSM/'
eng = matlab.engine.start_matlab()
gen_path(pth)