from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import PIL
import imageio
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):

  def __init__(self):
    super(Generator, self).__init__()
    self.conv1 = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
    self.conv1_bn = layers.BatchNormalization()
    self.conv2 = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()
    self.conv4 = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same')

  def call(self, inputs, training=True):
    """Run the model."""
    conv1 = self.conv1(inputs)
    conv1_bn = self.conv1_bn(conv1, training=training)
    conv1 = tf.nn.relu(conv1_bn)

    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv2 = tf.nn.relu(conv2_bn)

    conv3 = self.conv3(conv2)
    conv3_bn = self.conv3_bn(conv3, training=training)
    conv3 = tf.nn.relu(conv3_bn)

    conv4 = self.conv4(conv3)
    generated_data = tf.nn.sigmoid(conv4)

    return generated_data


class Discriminator(tf.keras.Model):

  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()

    self.conv4_base = layers.Conv2D(1, (3, 3))
    
    self.conv4_rot = layers.Dense(4, input_shape=(256, 256*3*3))

  def call(self, inputs, training=True, predict_rotation=False):
    conv1 = tf.nn.leaky_relu(self.conv1(inputs))
    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv3 = self.conv3(conv2_bn)
    conv3_bn = self.conv3_bn(conv3, training=training)
    if predict_rotation:
      conv3_flattened = tf.reshape(conv3_bn, (tf.shape(conv3_bn)[0], -1))
      rotation_class = self.conv4_rot(conv3_flattened)
      return rotation_class
    else:
      conv4_base = self.conv4_base(conv3_bn)
      discriminator_logits = tf.squeeze(conv4_base, axis=[1, 2])
      return discriminator_logits
