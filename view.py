# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:15:37 2019

@author: Sooram Kang

"""

import subprocess
import os

#directory = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\train_log9_filter많이\\out8\\'
directory = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\out\\'


#%%
epoch = 'test'

for index in range(10):
    path = directory + str(epoch) + '\\' + str(index) + '.binvox'
    args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\viewvox.exe " + path

    subprocess.call(args)
#%%
epoch = 144
index = 4
path = directory + str(epoch) + '\\' + str(index) + '.binvox'
args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\viewvox.exe " + path

subprocess.call(args)

#%%    
index = 239
path = directory + str(index) + '.binvox'
args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\viewvox.exe " + path

subprocess.call(args)

#%%

directory = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\'

data_list = os.listdir(directory + 'bed')

for i, data in enumerate(data_list):
    print(i)
    path = directory + 'bed\\' + data
    args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\binvox.exe " + path + " -d 64"

    subprocess.call(args)
    
    

    
import shutil

dest = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\chair_binvox'
for i, data in enumerate(data_list):
    if('.binvox' in data):
        shutil.copy2(directory+data, dest)
        print(i)

        
import numpy as np
import binvox


X = []

examples_dir = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\chair_binvox\\'
for example in os.listdir(examples_dir):
    if 'binvox' in example:
        with open(os.path.join(examples_dir, example), 'rb') as file:
            data = np.int32(binvox.read_as_3d_array(file).data)
#            padded_data = np.pad(data, 3, 'constant')
            X.append(data)

np.savez_compressed('modelnet10_chair3.npz',
                    X_train=X)