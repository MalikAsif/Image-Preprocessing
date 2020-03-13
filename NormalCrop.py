# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 23:39:08 2018

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
           if ".png" in file:
                PNGFile= file;
                print(PNGFile)
                path = new_path+'/'
                path_1= path+file;
                path_2= path+'a'+file;
                path_3= path+'b'+file;
                path_4= path+'c'+file;
                path_5= path+'d'+file;
                image = cv2.imread(filepath);
                height, width, channel= image.shape
                # crop from center point of image
                r_h =int(height/2);
                r_w  = int(width/2);
                x_min= r_w-100;
                x_max =r_w+100;
                y_min= r_h-100;
                y_max = r_h+100;
                # x-100 crop
                r_h1 = r_h;
                r_w1 =r_w-100
                x_min1= r_w1-100;
                x_max1 =r_w1+100;
                y_min1= r_h1-100;
                y_max1 = r_h1+100;
                #x+100 crop
                r_h2 = r_h;
                r_w2 =r_w+100
                x_min2= r_w2-100;
                x_max2 =r_w2+100;
                y_min2= r_h2-100;
                y_max2 = r_h2+100;
                # y-100 crop
                r_h3 = r_h-100;
                r_w3 =r_w
                x_min3= r_w3-100;
                x_max3 =r_w3+100;
                y_min3= r_h3-100;
                y_max3 = r_h3+100;
                # y+100 crop
                r_h4 = r_h+100;
                r_w4 =r_w
                x_min4= r_w4-100;
                x_max4 =r_w4+100;
                y_min4= r_h4-100;
                y_max4 = r_h4+100;
              
                number= number+1;
                roi_1= image[y_min:y_max,x_min:x_max]
                roi_2= image[y_min1:y_max1,x_min1:x_max1]
                roi_3= image[y_min2:y_max2,x_min2:x_max2]
                roi_4= image[y_min3:y_max3,x_min3:x_max3]
                roi_5= image[y_min3:y_max3,x_min3:x_max3]
           
                cv2.imwrite(path_1,roi_1)
                cv2.imwrite(path_2,roi_2)
                cv2.imwrite(path_3,roi_3)
                cv2.imwrite(path_4,roi_4)
                cv2.imwrite(path_5,roi_5)
                print("image saved")
                cv2.destroyAllWindows()
              
             
def gen_path(p):
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p + file
          for case in [doc for doc in os.listdir(path) if '.' not in doc]:
              SelectROI(path,case) 


pth = 'D:/BreastCancerThesis/DDSM/'
gen_path(pth)