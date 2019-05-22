# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:49:09 2019

@author: Sooram Kang

"""
import tensorflow as tf
from general_model import Model

class DCGAN(Model):
    def __init__(self, inputs, noises, config, is_training):
        self.inputs = inputs
        self.noises = noises
        self.config = config
        self.is_training = is_training
        
        net_G = Generator()
        net_D = Discriminator()
        
        G = net_G(noises, config, is_training)
        self.G = G
        D_real = net_D(inputs, is_training)
        D_fake = net_D(G, is_training, True)
        
        loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_real, labels=tf.ones_like(D_real)
        ))
        loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake, labels=tf.zeros_like(D_fake)
        ))
        
        loss_D = loss_D_real + loss_D_fake
        self.loss_D = loss_D

        
        loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake, labels=tf.ones_like(D_fake)
        ))
        self.loss_G = loss_G
        
        D_global_step = tf.Variable(0, trainable=False, name='D_global_step')
        G_global_step = tf.Variable(0, trainable=False, name='G_global_step')
        
        vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='discriminator')
        self.vars_D = vars_D
        vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='generator')
        self.vars_G = vars_G 
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_D = tf.train.AdamOptimizer(config['lr_D']).minimize(loss_D,
                var_list=vars_D, global_step=D_global_step)
            self.train_D = train_D
            train_G = tf.train.AdamOptimizer(config['lr_G']).minimize(loss_G,
                var_list=vars_G, global_step=G_global_step)
            self.train_G = train_G
    
        super(DCGAN, self).__init__(50)


def lrelu(x, leak=0.2):
    return tf.maximum(x, x * leak)

        
class Generator(object):
    def __call__(self, noise, config, is_training):
        with tf.variable_scope('generator'):
            output = tf.layers.dense(noise, 512*config["last"]*config["last"]*config["last"])
            output = tf.reshape(output, [-1, config["last"], config["last"], config["last"], 512])
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d_transpose(output, 256, [5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d_transpose(output, 128, [5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d_transpose(output, 64, [5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d_transpose(output, 32, [5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d_transpose(output, 1, [5, 5, 5], strides=(1, 1, 1), padding='SAME')
            output = tf.nn.sigmoid(output)
            return output
            


        
class Discriminator(object):        
    def __call__(self, inputs, is_training, reuse=None):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            output = tf.layers.conv3d(inputs, filters=32, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(output)
            output = tf.layers.conv3d(output, filters=64, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d(output, filters=128, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d(output, filters=256, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv3d(output, filters=512, kernel_size=[5, 5, 5], strides=(2, 2, 2), padding='SAME')
            output = lrelu(tf.layers.batch_normalization(output, training=is_training))
            flat = tf.contrib.layers.flatten(output)
            output = tf.layers.dense(flat, 1, activation=None)
            return output

