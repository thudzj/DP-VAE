from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt


import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from util.dataset import make_dataset
from util.layers import transposed_fully_connected
from util.metrics import cluster_acc, cluster_nmi

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar
import math

flags = tf.flags
tf.logging.set_verbosity(tf.logging.ERROR)
logging = tf.logging

flags.DEFINE_integer("batch_size", 200, "batch size") #128
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch") #1000
flags.DEFINE_integer("max_epoch", 1000, "max epoch") #100
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_integer("T", 20, "level of truncation")
flags.DEFINE_float("alpha_0", 1.0, "alpha_0 for the prior Beta distribution")
flags.DEFINE_float("beta_0", 1.0, "beta_0 for the prior Beta distribution")
flags.DEFINE_float("sigma2", 1.0, "covariance for the mixtured gaussian components")

FLAGS = flags.FLAGS

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
            transposed_fully_connected(in_size = FLAGS.T, activation_fn=None)).tensor

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

def get_S_loss(alpha, beta, mean_x, logcov_x, mean_eta, logcov_eta, sigma2, epsilon=1e-8):
    mean_x_pad = tf.expand_dims(mean_x, 1)
    logcov_x_pad = tf.expand_dims(logcov_x, 1)
    mean_eta_pad = tf.expand_dims(mean_eta, 0)
    logcov_eta_pad = tf.expand_dims(logcov_eta, 0)
    S = 0.5 * tf.reduce_sum( \
            1 + logcov_x_pad - math.log(sigma2) \
            - (tf.exp(logcov_x_pad) + tf.exp(logcov_eta_pad) + tf.square(mean_x_pad - mean_eta_pad)) / sigma2 , 2 \
        ) \
        + tf.digamma(alpha) - tf.digamma(alpha + beta) + tf.cumsum(tf.digamma(beta) - tf.digamma(alpha + beta), exclusive=True)

    assignments = tf.argmax(S, axis=1)
    S_max = tf.reduce_max(S, axis=1)
    S_loss = -tf.reduce_sum(S_max) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S - tf.expand_dims(S_max, 1)), axis = 1) + epsilon))
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
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = []
    for grad, var in grads_and_vars:
        if grad is not None:
            clipped_grads_and_vars.append((tf.clip_by_value(grad, -1., 1.), var))
        print(grad, var)
    train = optimizer.apply_gradients(clipped_grads_and_vars)
    #train = pt.apply_optimizer(optimizer, losses=[loss])
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0
            loss1 = 0.0
            loss2 = 0.0
            loss3 = 0.0
            loss4 = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, _ = mnist_full.next_batch(FLAGS.batch_size)
                _, loss_value, loss1_, loss2_, loss3_, loss4_ = sess.run([train, loss, kl_alphabeta, kl_eta, S_loss, rec_loss], {input_tensor: x})
                training_loss += loss_value
                loss1 += loss1_
                loss2 += loss2_
                loss3 += loss3_
                loss4 += loss4_

            training_loss = training_loss / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss1 = loss1 / FLAGS.updates_per_epoch / float(N)
            loss2 = loss2 / FLAGS.updates_per_epoch / float(N)
            loss3 = loss3 / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss4 = loss4 / FLAGS.updates_per_epoch / FLAGS.batch_size

            mnist_full._index_in_epoch = 0
            num_batches = int(N / FLAGS.batch_size)
            labels_pred = np.zeros(num_batches * FLAGS.batch_size)
            labels = np.zeros(num_batches * FLAGS.batch_size)
            for i in range(num_batches):
                x, y = mnist_full.next_batch(FLAGS.batch_size)
                y_p = sess.run(assignments, {input_tensor: x})
                labels[i * FLAGS.batch_size : (i + 1) * FLAGS.batch_size] = y
                labels_pred[i * FLAGS.batch_size : (i + 1) * FLAGS.batch_size] = y_p
            print("Loss: %f, kl_alphabeta: %f, kl_eta: %f, S_loss: %f, rec_loss: %f, acc: %f, nmi: %f" 
                % (training_loss, loss1, loss2, loss3, loss4, cluster_acc(labels_pred, labels), cluster_nmi(labels_pred, labels)))

            imgs = sess.run(sampled_tensor)
            for k in range(FLAGS.batch_size):
                imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)

                imsave(os.path.join(imgs_folder, '%d.png') % k,
                       imgs[k].reshape(28, 28))
