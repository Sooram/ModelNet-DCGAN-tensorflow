# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:48:10 2019

@author: Sooram Kang

"""
import subprocess
import os


directory = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\bed\\'
data_list = os.listdir(directory)

for i, data in enumerate(data_list):
    print(i)
    path = directory + data
    args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\binvox.exe " + path + " -d 64"    #(64,64,64)

    subprocess.call(args)
    
#%%
""" copy .binvox files into 'dest' dir """    
import shutil

dest = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\chair_binvox'
for i, data in enumerate(data_list):
    if('.binvox' in data):
        shutil.copy2(directory+data, dest)
        print(i)
    