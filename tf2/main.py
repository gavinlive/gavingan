# Code from : https://github.com/ilguyi/gans.tensorflow.v2/blob/master/tf.v2/01.dcgan.ipynb
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import PIL
import imageio
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers
tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"]="0"




# Training Flags (hyperparameter configuration)
train_dir = 'train/01.dcgan/exp1/'
max_epochs = 20
save_epochs = 10
print_steps = 100
batch_size = 256
learning_rate_D = 0.001
learning_rate_G = 0.005
k = 1 # the number of step of learning D before learning G
num_examples_to_generate = 16
noise_dim = 100











# Load training and eval data from tf.keras
(train_data, train_labels), _ = \
    tf.keras.datasets.mnist.load_data()

train_data = train_data.reshape(-1, 28, 28, 1).astype('float32')
train_data = train_data / 255.
train_labels = np.asarray(train_labels, dtype=np.int32)







tf.set_random_seed(219)

# for train
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.shuffle(buffer_size = 60000)
train_dataset = train_dataset.batch(batch_size = batch_size)
print(train_dataset)









class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.conv1 = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
    self.conv1_bn = layers.BatchNormalization()
    self.conv2 = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.conv3_bn = layers.BatchNormalization()
    self.conv4 = layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same')

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
    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)
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





generator = Generator()
discriminator = Discriminator()




# Defun for performance boost
generator.call = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)





def GANLoss(logits, is_real=True):
  """Computes standard GAN loss between `logits` and `labels`.

  Args:
    logits (`1-rank Tensor`): logits.
    is_real (`bool`): True means `1` labeling, False means `0` labeling.

  Returns:
    loss (`0-randk Tensor): the standard GAN loss value. (binary_cross_entropy)
  """
  if is_real:
    labels = tf.ones_like(logits)
  else:
    labels = tf.zeros_like(logits)

  return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                         logits=logits)




def discriminator_loss(real_logits, fake_logits):
    # losses of real with label "1"
    real_loss = GANLoss(logits=real_logits, is_real=True)
    # losses of fake with label "0"
    fake_loss = GANLoss(logits=fake_logits, is_real=False)

    total_loss = real_loss + fake_loss

    return total_loss





def generator_loss(fake_logits):
  # losses of Generator with label "1" that used to fool the Discriminator
  return GANLoss(logits=fake_logits, is_real=True)







#discriminator_optimizer = tf.train.AdamOptimizer(learning_rate_D, beta1=0.5)
discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate_D)
generator_optimizer = tf.train.AdamOptimizer(learning_rate_G, beta1=0.5)






'''
Checkpointing
'''
#checkpoint_dir = './training_checkpoints'
checkpoint_dir = train_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)




'''
Training
'''
# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
random_vector_for_generation = tf.random_normal([num_examples_to_generate, 1, 1, noise_dim])

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()



def print_or_save_sample_images(sample_data, max_print=num_examples_to_generate, is_save=False, epoch=None):
  print_images = sample_data[:max_print,:]
  print_images = print_images.reshape([max_print, 28, 28])
  print_images = print_images.swapaxes(0, 1)
  print_images = print_images.reshape([28, max_print * 28])

  plt.figure(figsize=(max_print, 1))
  plt.axis('off')
  plt.imshow(print_images, cmap='gray')

  if is_save and epoch is not None:
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()




'''
Training Loop
'''
tf.logging.info('Start Session.')
#global_step = tf.train.get_or_create_global_step()
step = 0
for epoch in range(max_epochs):

  for images in train_dataset:
    start_time = time.time()

    # generating noise from a uniform distribution
    noise = tf.random_normal([batch_size, 1, 1, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_logits = discriminator(images, training=True)
      fake_logits = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_logits)
      disc_loss = discriminator_loss(real_logits, fake_logits)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

    epochs = step * batch_size / float(len(train_data))
    duration = time.time() - start_time

    if step % print_steps == 0:
      display.clear_output(wait=True)
      examples_per_sec = batch_size / float(duration)
      print("Epochs: {:.2f} global_step: {} loss_D: {:.3f} loss_G: {:.3f} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
                epochs, step, disc_loss, gen_loss, examples_per_sec, duration))
      sample_data = generator(random_vector_for_generation, training=False)
      print_or_save_sample_images(sample_data.numpy())

    step += 1

  if epoch % 1 == 0:
    display.clear_output(wait=True)
    print("This images are saved at {} epoch".format(epoch+1))
    sample_data = generator(random_vector_for_generation, training=False)
    print_or_save_sample_images(sample_data.numpy(), is_save=True, epoch=epoch+1)

  # saving (checkpoint) the model every save_epochs
  if (epoch + 1) % save_epochs == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)




'''
Final Epoch
'''

# generating after the final epoch
display.clear_output(wait=True)
sample_data = generator(random_vector_for_generation, training=False)
print_or_save_sample_images(sample_data.numpy(), is_save=True, epoch=max_epochs)





'''
Restore Checkpoint
'''

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



'''
Image Epoch Number
'''


def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(max_epochs)


'''
Generate GIFs
'''
with imageio.get_writer('dcgan.gif', mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

# this is a hack to display the gif inside the notebook
os.system('cp dcgan.gif dcgan.gif.png')


display.Image(filename="dcgan.gif.png")
