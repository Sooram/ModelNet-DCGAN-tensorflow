# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:55:37 2019

@author: Sooram Kang

"""
import os
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import binvox
from dcgan import DCGAN

config = {
    "x": 38,
    "y": 38,
    "z": 38,
    "img_C": 1,
    "lr_D": 0.000025,#1e-4,
    "lr_G": 0.0003
}

total_epoch = 1000
batch_size = 64
n_noise = 500

log_dir = './train_log5/'
saving_cycle = 1
start_epoch = 0

def get_noise(batch_size, n_noise):                                                                                 
    return np.random.uniform(-1.0, 1.0, size=[batch_size, n_noise])

def get_moving_noise(batch_size, n_noise):
    assert batch_size > 0
 
    noise_list = []
    base_noise = np.random.uniform(-1.0, 1.0, size=[n_noise])
    end_noise = np.random.uniform(-1.0, 1.0, size=[n_noise])
 
    step = (end_noise - base_noise) / batch_size
    noise = np.copy(base_noise)
    for _ in range(batch_size - 1):
        noise_list.append(noise)
        noise = noise + step
    noise_list.append(end_noise)
    
    return noise_list    
    
def load_middle_point_model(model_path):
    # model_path example: 'C:\\Users\\CHANG\\PycharmProjects\\p260\\model/model.ckpt-46'

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "0" #gpu_num

    sess = tf.Session(config=c)
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph(model_path + '.meta')
    new_saver.restore(sess, model_path)
    tf.get_default_graph().as_graph_def()

    return sess
    
def save_binvox(filename, data):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(filename, 'wb') as f:
        model.write(f)

""" load data """ 
#data = np.load('modelnet10.npz')
#X_train = data['X_train'][621:1510] # chair


data = np.load('modelnet10_chair.npz')
X_train = data['X_train'] # chair
#X_train, Y_train = shuffle(data['X_train'], data['y_train'])
#X_test, Y_test = shuffle(data['X_test'], data['y_test'])

#X_train.shape   # (3991, 30, 30, 30)
#X_test.shape    # (908, 30, 30, 30)


""" build model """
X = tf.placeholder(tf.float32, [None, config['x'], config['y'], config['z'], 1])
Z = tf.placeholder(tf.float32, [None, n_noise])
is_training = tf.placeholder(tf.bool)

model = DCGAN(X, Z, config, is_training)


""" open session """
c = tf.ConfigProto()
c.gpu_options.visible_device_list = "0"

sess = tf.Session(config=c)
sess.run(tf.global_variables_initializer())

#model.load(sess, log_dir)  


#%%
""" train1 """
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
total_batch = int(len(X_train)/batch_size) + 1
 
for epoch in range(start_epoch,total_epoch):
    
    for i in range(total_batch):
        batch_xs = X_train[i*batch_size: (i+1)*batch_size]
        batch_xs = batch_xs.reshape(-1, config['x'], config['y'], config['z'], 1)
        noise = get_noise(batch_size, n_noise)
 
        _, loss_val_D = sess.run([model.train_D, model.loss_D],
            feed_dict={X: batch_xs, Z: noise, is_training: True})
        _, loss_val_G = sess.run([model.train_G, model.loss_G],
            feed_dict={X: batch_xs, Z: noise, is_training: True})
 
 
    print('Epoch:', '%04d' % epoch,
        'D loss: {:.4}'.format(loss_val_D),
        'G loss: {:.4}'.format(loss_val_G))
    
    with open(log_dir + "training_loss.txt", "a+") as file:
            file.write("Epoch: %d\t LossD: %f\t LossG: %f\n" % (epoch, loss_val_D, loss_val_G))

    if epoch == 0 or (epoch + 1) % 5 == 0:    
#    if(epoch % saving_cycle == 0):   
        model.save(sess, log_dir, epoch)
    
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(model.G, feed_dict={Z: noise, is_training: False})
        print(samples.min(), samples.max())
#        test_noise = get_moving_noise(sample_size, n_noise)
#        test_samples = sess.run(model.G, feed_dict={Z: test_noise, is_training: False})
        path = "out5/{}/".format(epoch)
        if not os.path.exists(path): os.makedirs(path)
        
        for i, data in enumerate(samples):
            save_binvox(path + "{}.binvox".format(i), data[:, :, :, 0] > 0.9)


#%%
""" test """
#model.load(sess, log_dir) 
#
#epoch = 'test'
#sample_size = 10
#noise = get_noise(sample_size, n_noise)
#samples = sess.run(model.G, feed_dict={Z: noise, is_training: False})
##test_noise = get_moving_noise(sample_size, n_noise)
##test_samples = sess.run(model.G, feed_dict={Z: test_noise, is_training: False})
#path = "out4/{}/".format(epoch)
#if not os.path.exists(path): os.makedirs(path)
#
#for i, data in enumerate(samples):
#    save_binvox(path + "{}.binvox".format(i), data[:, :, :, 0] > 0.9)            




