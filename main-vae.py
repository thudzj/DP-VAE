'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
from prettytensor import layers
import collections
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from util.dataset import make_dataset

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar
import math

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 10, "number of updates per epoch") #1000
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_integer("L", 20, "level of truncation")
flags.DEFINE_float("alpha_0", 1.0, "alpha_0 for the prior Beta distribution")
flags.DEFINE_float("beta_0", 1.0, "beta_0 for the prior Beta distribution")
flags.DEFINE_float("sigma2", 1.0, "covariance for the mixtured gaussian components")

FLAGS = flags.FLAGS

@pt.Register(assign_defaults=('activation_fn', 'l2loss',
                                        'parameter_modifier', 'phase'))
class transposed_fully_connected(pt.VarStoreMethod):

  def __call__(self,
               input_layer,
               in_size,
               activation_fn=None,
               l2loss=None,
               weights=None,
               bias=tf.zeros_initializer,
               transpose_weights=False,
               phase=pt.Phase.train,
               parameter_modifier=None,
               name="transposed_fully_connected"):
    in_size = in_size
    size = input_layer.shape[0]
    books = input_layer.bookkeeper
    if weights is None:
      weights = layers.he_init(in_size, size, activation_fn)

    dtype = input_layer.tensor.dtype
    weight_shape = [size, in_size] if transpose_weights else [in_size, size]

    params = self.variable('transposed_fully_connected_weights', weight_shape, weights, dt=dtype)
    y = tf.matmul(params, input_layer, transpose_b=transpose_weights)
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += self.variable('transposed_fully_connected_bias', [input_layer.shape[1]], bias, dt=dtype)

    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)


def encoder(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        
    '''
    mid_features = (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten())
    x_attributes = (mid_features.fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor
    clusters_attributes = (mid_features.
            fully_connected(FLAGS.hidden_size * 2 + 2).
            transposed_fully_connected(in_size = FLAGS.L, activation_fn=None)).tensor

    mean_eta = clusters_attributes[:, :FLAGS.hidden_size]
    logcov_eta = clusters_attributes[:, FLAGS.hidden_size:FLAGS.hidden_size*2]
    alpha = tf.squeeze(tf.exp(clusters_attributes[:, FLAGS.hidden_size*2]))
    beta = tf.squeeze(tf.exp(clusters_attributes[:, FLAGS.hidden_size*2 + 1]))
    return x_attributes, mean_eta, logcov_eta, alpha, beta


def decoder(input_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from 
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    if input_tensor is None:
        mean = None
        logcov = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.hidden_size]
        logcov = input_tensor[:, FLAGS.hidden_size:]
        stddev = tf.sqrt(tf.exp(logcov))
        input_sample = mean + epsilon * stddev
    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()
            # fully_connected(2000, name='decoder_l1', activation_fn=tf.nn.relu).
            # fully_connected(500, name='decoder_l2', activation_fn=tf.nn.relu).
            # fully_connected(500, name='decoder_l3', activation_fn=tf.nn.relu).
            # fully_connected(784, name='decoder_l4', activation_fn=tf.nn.sigmoid)
            ).tensor, mean, logcov


def kl_Gaussian(mean, logcov, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean: 
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.exp(logcov) - logcov - 1.0))

def kl_Beta(alpha, beta, alpha_0, beta_0):
    return tf.reduce_sum(math.lgamma(alpha_0) + math.lgamma(beta_0) - math.lgamma(alpha_0+beta_0)
                + tf.lgamma(alpha + beta) - tf.lgamma(alpha) - tf.lgamma(beta)
                + (alpha - alpha_0) * tf.digamma(alpha) + (beta - beta_0) * tf.digamma(beta)
                - (alpha + beta - alpha_0 - beta_0) * tf.digamma(alpha + beta)
        )

def get_S_loss(alpha, beta, mean_x, logcov_x, mean_eta, logcov_eta, sigma2):
    mean_x_pad = tf.expand_dims(mean_x, 1)
    logcov_x_pad = tf.expand_dims(logcov_x, 1)
    mean_eta_pad = tf.expand_dims(mean_eta, 0)
    logcov_eta_pad = tf.expand_dims(logcov_eta, 0)
    S1 = 0.5 * tf.reduce_sum(1 + logcov_x_pad - math.log(sigma2) - \
        (tf.exp(logcov_x_pad) + tf.exp(logcov_eta_pad) + tf.square(mean_x_pad - mean_eta_pad)) / sigma2 , 2)
    S2 = tf.digamma(alpha) - tf.digamma(alpha + beta) + tf.cumsum(tf.digamma(beta) - tf.digamma(alpha + beta), exclusive=True)
    S = S1 + S2
    assignments = tf.argmax(S)
    S_loss = -tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S), axis = 1)))
    return assignments, S_loss

def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder 
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''
    return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                         (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=False)
    images = np.concatenate((mnist.validation.images, mnist.train.images, mnist.test.images))
    labels = np.concatenate((mnist.validation.labels, mnist.train.labels, mnist.test.labels))
    mnist_full = make_dataset(images, labels)
    N = mnist_full.num_examples
    print(N)

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                x_attributes, mean_eta, logcov_eta, alpha, beta = encoder(input_tensor)
            with tf.variable_scope("decoder") as scope:
                output_tensor, mean_x, logcov_x = decoder(x_attributes)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("decoder", reuse=True) as scope:
                sampled_tensor, _, _ = decoder()

    kl_eta = kl_Gaussian(mean_eta, logcov_eta)
    kl_alphabeta = kl_Beta(alpha, beta, FLAGS.alpha_0, FLAGS.beta_0)
    assignments, S_loss = get_S_loss(alpha, beta, mean_x, logcov_x, mean_eta, logcov_eta, FLAGS.sigma2)
    rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

    loss = (kl_eta + kl_alphabeta) / float(N) * float(FLAGS.batch_size) + rec_loss + S_loss
    #loss = rec_loss + kl_Gaussian(mean_x, logcov_x)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.initialize_all_variables()
    for var in tf.trainable_variables():
        print(var.name)

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, _ = mnist_full.next_batch(FLAGS.batch_size)
                _, loss_value, loss1, loss2, loss3 = sess.run([train, loss, kl_eta, kl_alphabeta, rec_loss], {input_tensor: x})
                training_loss += loss_value

            training_loss = training_loss / \
                (FLAGS.updates_per_epoch * FLAGS.batch_size)

            print(loss1, loss2, loss3)
            print("Loss %f  " % training_loss)

            imgs = sess.run(sampled_tensor)
            for k in range(FLAGS.batch_size):
                imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)

                imsave(os.path.join(imgs_folder, '%d.png') % k,
                       imgs[k].reshape(28, 28))
