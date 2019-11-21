import tensorflow as tf

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

  return tf.nn.sigmoid_cross_entropy_with_logits(labels,logits)




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
