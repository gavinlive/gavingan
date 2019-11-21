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
#%matplotlib inline

import PIL
import imageio
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers
#tf.enable_eager_execution()

#tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from layers import Generator, Discriminator, RotationDiscriminator_temp
from loss import GANLoss, discriminator_loss, generator_loss

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
    tf.keras.datasets.cifar10.load_data()

train_data = train_data.reshape(-1, 32, 32, 3).astype('float32')
train_data = train_data / 255.
train_labels = np.asarray(train_labels, dtype=np.int32)



tf.random.set_seed(219)
operation_seed = None

# for train
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.shuffle(buffer_size = 60000)
train_dataset = train_dataset.batch(batch_size = batch_size)

print(train_dataset)


generator = Generator()
discriminator = RotationDiscriminator_temp()




# Defun for performance boost
#generator.call = tf.contrib.eager.defun(generator.call)
#discriminator.call = tf.contrib.eager.defun(discriminator.call)













#discriminator_optimizer = tf.train.AdamOptimizer(learning_rate_D, beta1=0.5)
#discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate_D)
#generator_optimizer = tf.train.AdamOptimizer(learning_rate_G, beta1=0.5)

discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate_D)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate_G, beta_1=0.5)





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
random_vector_for_generation = tf.random.normal([num_examples_to_generate, 1, 1, noise_dim])

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



def print_or_save_sample_images(sample_data, max_print=num_examples_to_generate, is_save=False, epoch=None, prefix=""):
  print_images = sample_data[:max_print,:]
  print_images = print_images.reshape([max_print, 32, 32, 3])
  print_images = print_images.swapaxes(0, 1)
  print_images = print_images.reshape([32, max_print * 32, 3])

  plt.figure(figsize=(max_print, 1))
  plt.axis('off')
  plt.imshow(print_images, cmap='gray')

  if is_save and epoch is not None:
    plt.savefig(prefix+'image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()




'''
Training Loop
'''
#f.logging.info('Start Session.')
#global_step = tf.train.get_or_create_global_step()
step = 0
for epoch in range(max_epochs):

  for images in train_dataset:
    start_time = time.time()

    # generating noise from a uniform distribution
    noise = tf.random.normal([batch_size, 1, 1, noise_dim], seed=operation_seed)

    # Generate the Affine Transforms
    #translation = tf.random.uniform([batch_size, 2, 1, 1], minval=-10, maxval=10, seed=operation_seed)
    #rotation = tf.random.uniform([batch_size, 1, 1, 1,], minval=-np.pi, maxval=np.pi)
    #shear = tf.random.uniform([batch_size, 1, 1, 1,], minval=-np.pi, maxval=np.pi)
    affine_parameters = tf.random.uniform([batch_size, 6], minval=-2, maxval=2, seed=operation_seed)
    affine_parameters = tf.concat([affine_parameters, tf.zeros([batch_size, 2])], axis=1)


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_logits, transformed_real_images = discriminator(images, affine_parameters, training=True)
      fake_logits, transformed_fake_images = discriminator(generated_images, affine_parameters, training=True)

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
      print(duration)
      examples_per_sec = batch_size / float(duration)
      print("Epochs: {:.2f} global_step: {} loss_D: {:.3f} loss_G: {:.3f} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
                epochs,
                step,
                disc_loss.numpy(),
                gen_loss.numpy(),
                examples_per_sec,
                duration))
      sample_data = generator(random_vector_for_generation, training=False)
      print_or_save_sample_images(sample_data.numpy())

    step += 1

  if epoch % 1 == 0:
    display.clear_output(wait=True)
    print("This images are saved at {} epoch".format(epoch+1))
    sample_data = generator(random_vector_for_generation, training=False)
    print_or_save_sample_images(sample_data.numpy(), is_save=True, epoch=epoch+1)
    print_or_save_sample_images(transformed_real_images.numpy(), is_save=True, epoch=epoch+1, prefix="REAL_TRANSFORMED_")

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
