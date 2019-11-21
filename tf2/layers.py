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


class ProjectiveTransformer(tf.keras.layers.Layer):
  def __init__(self, name=None):
    # c0, c1 = 0
    super(ProjectiveTransformer, self).__init__()
    #self.name = name

  def build(self, input_shape):
    pass

  def call(self, input, transform_matrix):
    '''
    transform matrix is a pose tensor composed of vectors (1 x 8). Size N x 8,
    where N is the batch size
    OR transform matrix is 1 x 8 vector

    Vector 1x8 = [a0, a1, a2, b0, b1, b2, c0, c1]
    maps the output point (x, y) to a transformed input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), where k = c0 x + c1 y + 1
    '''
    transformed = tfa.image.transform(input, transform_matrix, name=None)
    return transformed


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
    self.conv4 = layers.Conv2D(1, (3, 3))

  def call(self, inputs, training=True):
    conv1 = tf.nn.leaky_relu(self.conv1(inputs))
    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv3 = self.conv3(conv2_bn)
    conv3_bn = self.conv3_bn(conv3, training=training)
    conv4 = self.conv4(conv3_bn)
    discriminator_logits = tf.squeeze(conv4, axis=[1, 2])

    return discriminator_logits


class RotationDiscriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()
    self.conv4 = layers.Conv2D(1, (3, 3))

  def call(self, inputs, affine_parameters=None, training=True):
    if affine_parameters is not None:
        print("Not Implemented yet")
    conv1 = tf.nn.leaky_relu(self.conv1(inputs))
    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv3 = self.conv3(conv2_bn)
    conv3_bn = self.conv3_bn(conv3, training=training)
    conv4 = self.conv4(conv3_bn)
    discriminator_logits = tf.squeeze(conv4, axis=[1, 2])

    return discriminator_logits

class RotationDiscriminator_temp(tf.keras.Model):
  def __init__(self):
    super(RotationDiscriminator_temp, self).__init__()
    self.conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()
    self.conv4 = layers.Conv2D(1, (3, 3))
    self.transformer = ProjectiveTransformer()

  def call(self, inputs, affine_parameters=None, training=True):
    if affine_parameters is not None:
        print("Not Implemented yet")
    conv1 = tf.nn.leaky_relu(self.conv1(inputs))
    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv3 = self.conv3(conv2_bn)
    conv3_bn = self.conv3_bn(conv3, training=training)
    conv4 = self.conv4(conv3_bn)
    discriminator_logits = tf.squeeze(conv4, axis=[1, 2])

    return discriminator_logits, self.transformer(inputs, affine_parameters)


class AffineDiscriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()
    self.conv4 = layers.Conv2D(1, (3, 3))

  def call(self, inputs, training=True):
    conv1 = tf.nn.leaky_relu(self.conv1(inputs))
    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv3 = self.conv3(conv2_bn)
    conv3_bn = self.conv3_bn(conv3, training=training)
    conv4 = self.conv4(conv3_bn)
    discriminator_logits = tf.squeeze(conv4, axis=[1, 2])
    #affine_params = layers.


    return discriminator_logits
