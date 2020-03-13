# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:58:13 2018

@author: Asif
"""

import os
import subprocess

def DeleteExtraFiles(path, case):
     new_path = path + '\\' + case
     for file in [doc for doc in os.listdir(new_path) if doc.endswith(".1") or doc.endswith(".pnm") or doc.endswith(".LJPEG")]:
           filepath= new_path+'\\'+file
           if os.path.exists(filepath):
               os.remove(filepath)
             
def gen_path(p):
    file_list1 = []
    for file in [doc for doc in os.listdir(p) if '.' not in doc]:
          path = p + "\\" + file
          for case in [doc for doc in os.listdir(path) if '.' not in doc]:
              DeleteExtraFiles(path,case) 
               
              
                
               
pth = 'D:\\BreastCancerThesis\\del'
gen_path(pth)