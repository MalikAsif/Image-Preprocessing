# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:20:38 2018

@author: Asif
"""


import os
import subprocess
import time

def Rename(path, case):
     new_path = path + '\\' + case
     for file in [doc for doc in os.listdir(new_path) ]:
           filepath= new_path+'\\'+file
           if ".LJPEG" in file:      
               newfile= file.split('.LJPEG')
               newfile= newfile[0]+'.png'
               newpath=new_path+'\\'+newfile
               print(newfile)
               os.rename(filepath,newpath)
           #if os.path.exists(filepath):
             #  os.remove(filepath)
             
def gen_path(p):
    file_list1 = []
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p + "\\" + file
          for case in [doc for doc in os.listdir(path) if '.' not in doc]:
              Rename(path,case) 
               
              
                
               
pth = 'D:\\BreastCancerThesis\\DDSM\\'
gen_path(pth)