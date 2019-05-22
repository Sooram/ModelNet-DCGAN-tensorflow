# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:45:36 2019

@author: Sooram Kang

"""
import os
import numpy as np
import binvox


X = []

examples_dir = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\bed\\bed_binvox\\'
for example in os.listdir(examples_dir):
    if 'binvox' in example:
        with open(os.path.join(examples_dir, example), 'rb') as file:
            data = np.int32(binvox.read_as_3d_array(file).data)
#            padded_data = np.pad(data, 3, 'constant')
            X.append(data)

np.savez_compressed('modelnet10_bed.npz',
                    X_train=X)