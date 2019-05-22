# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:15:37 2019

@author: Sooram Kang

"""

import subprocess

directory = 'C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\DCGAN\\modelnet\\bed\\out\\'

#%%
""" visualize all the files in one epoch """
epoch = 'test3'

for index in range(10):
    path = directory + str(epoch) + '\\' + str(index) + '.binvox'
    args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\viewvox.exe " + path

    subprocess.call(args)
#%%
""" visualize a specific file in one epoch """
epoch = 144
index = 4
path = directory + str(epoch) + '\\' + str(index) + '.binvox'
args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\viewvox.exe " + path

subprocess.call(args)

#%%  
""" visualize a specific file """  
index = 104
path = directory + str(index) + '.binvox'
args = "C:\\Users\\CHANG\\PycharmProjects\\3dCNN\\viewvox.exe " + path

subprocess.call(args)


#%%
""" screen shot """
import PIL.ImageGrab

im = PIL.ImageGrab.grab()     
im.show()

