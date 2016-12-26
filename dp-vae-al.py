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
from sklearn.mixture import BayesianGaussianMixture

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("batch_size", 350, "batch size") #128
flags.DEFINE_integer("updates_per_epoch", 600, "number of updates per epoch") #1000
flags.DEFINE_integer("max_epoch", 300, "max epoch") #100
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_integer("T", 10, "level of truncation")
flags.DEFINE_float("lam", 1.0, "weight of the regularizer")
FLAGS = flags.FLAGS

def encoder(input_tensor):
    '''Create encoder network. 32 64 128

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        
    '''
    mid_features = (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges='VALID').
            dropout(0.9).
            flatten())
    x_attributes = (mid_features.fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor
    mean_x = x_attributes[:, :FLAGS.hidden_size]
    logcov_x = x_attributes[:, FLAGS.hidden_size:]

    return mean_x, logcov_x


def decoder(mean=None, logcov = None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from 
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    if mean is None and logcov is None:
        mean = None
        logcov = None
        stddev = None
        input_sample = epsilon
    else:
        stddev = tf.sqrt(tf.exp(logcov))
        input_sample = mean + epsilon * stddev
    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d(3, 256, edges='VALID').
            deconv2d(5, 128, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()
            # fully_connected(2000, name='decoder_l1', activation_fn=tf.nn.relu).
            # fully_connected(500, name='decoder_l2', activation_fn=tf.nn.relu).
            # fully_connected(500, name='decoder_l3', activation_fn=tf.nn.relu).
            # fully_connected(784, name='decoder_l4', activation_fn=tf.nn.sigmoid)
            ).tensor

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

def kl_Gaussian(mean_x, logcov_x, mu_tensor, pre_tensor, epsilon=1e-8):
    mean_x_pad = tf.expand_dims(mean_x, 1)
    mu_tensor_pad = tf.expand_dims(mu_tensor, 0)
    logcov_x_pad = tf.expand_dims(logcov_x, 1)
    pre_tensor_pad = tf.expand_dims(pre_tensor, 0)
    return 0.5 * tf.reduce_sum((pre_tensor_pad*tf.exp(logcov_x_pad) + pre_tensor_pad * tf.square(mean_x_pad-mu_tensor_pad) 
        - 1 - tf.log(pre_tensor_pad) - logcov_x_pad), 2)

def get_regularizer(S):
    return tf.reduce_sum(tf.nn.softmax(S) * tf.nn.log_softmax(S))

def get_S_loss(mean_x, logcov_x, pi_tensor, mu_tensor, pre_tensor, epsilon=1e-8):
    S = -kl_Gaussian(mean_x, logcov_x, mu_tensor, pre_tensor) + tf.log(pi_tensor + epsilon)
    reg = get_regularizer(S) * FLAGS.lam
 
    assignments = tf.argmax(S, dimension=1)
    S_max = tf.reduce_max(S, reduction_indices=1)
    S_loss = -tf.reduce_sum(S_max) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S - tf.expand_dims(S_max, 1)), reduction_indices = 1) + epsilon))
    #S_loss = - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S), reduction_indices = 1) + epsilon))
    return assignments, S_loss, reg


if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=False)
    images = np.concatenate((mnist.validation.images, mnist.train.images, mnist.test.images))
    labels = np.concatenate((mnist.validation.labels, mnist.train.labels, mnist.test.labels))
    mnist_full = make_dataset(images, labels)
    N = mnist_full.num_examples
    pis = np.ones((FLAGS.T)) / FLAGS.T
    mus = np.zeros((FLAGS.T, FLAGS.hidden_size))
    precisions = np.ones((FLAGS.T, FLAGS.hidden_size))

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])
    pi_tensor = tf.placeholder(tf.float32, [FLAGS.T])
    mu_tensor = tf.placeholder(tf.float32, [FLAGS.T, FLAGS.hidden_size])
    pre_tensor = tf.placeholder(tf.float32, [FLAGS.T, FLAGS.hidden_size])

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                mean_x, logcov_x = encoder(input_tensor)
            with tf.variable_scope("decoder") as scope:
                output_tensor = decoder(mean_x, logcov_x)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("decoder", reuse=True) as scope:
                sampled_tensor = decoder()

    assignments, S_loss, reg = get_S_loss(mean_x, logcov_x, pi_tensor, mu_tensor, pre_tensor)
    rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

    loss = rec_loss + S_loss
    vae_loss = rec_loss + tf.reduce_sum(0.5 * (tf.square(mean_x) + tf.exp(logcov_x) - logcov_x - 1.0))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    # grads_and_vars = optimizer.compute_gradients(loss)
    # clipped_grads_and_vars = []
    # for grad, var in grads_and_vars:
        # if grad is not None:
            # clipped_grads_and_vars.append((tf.clip_by_value(grad, -1., 1.), var))
        # #print(grad, var)
    # train = optimizer.apply_gradients(clipped_grads_and_vars)

    train = pt.apply_optimizer(optimizer, losses=[loss])
    train_reg = pt.apply_optimizer(optimizer, losses=[loss + reg])
    train_vae = pt.apply_optimizer(optimizer, losses=[vae_loss])
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        
        print("Pre-training the vae model")
        #gmm = GaussianMixture(n_components = FLAGS.T, covariance_type="diag", warm_start=True)
        for epoch in range(0):
            training_loss = 0.0
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, _ = mnist_full.next_batch(FLAGS.batch_size)
                _, loss_value = sess.run([train_vae, vae_loss], {input_tensor: x})
                training_loss += loss_value

            training_loss = training_loss / \
                (FLAGS.updates_per_epoch * FLAGS.batch_size)

            print("Loss %f" % training_loss)

        print("Training the dp-vae model")
        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0
            loss1 = 0.0
            loss2 = 0.0
            loss3 = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, y = mnist_full.next_batch(FLAGS.batch_size)
                if epoch < 1000:
                  _, y_p, loss_value, loss1_, loss2_, loss3_ = sess.run([train, assignments, loss, S_loss, rec_loss, reg],
                    {input_tensor: x, mu_tensor: mus, pi_tensor: pis, pre_tensor: precisions})                  
                else:
                  _, y_p, loss_value, loss1_, loss2_, loss3_ = sess.run([train_reg, assignments, loss, S_loss, rec_loss, reg], 
                    {input_tensor: x, mu_tensor: mus, pi_tensor: pis, pre_tensor: precisions})
                training_loss += loss_value
                loss1 += loss1_
                loss2 += loss2_
                loss3 += loss3_

            training_loss = training_loss / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss1 = loss1 / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss2 = loss2 / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss3 = loss3 / FLAGS.updates_per_epoch / FLAGS.batch_size
                
            print("Loss: %f, S_loss: %f, rec_loss: %f, reg: %f" 
                % (training_loss, loss1, loss2, loss3))

            all_batches = int(N / FLAGS.batch_size)
            features = np.zeros((N, FLAGS.hidden_size))
            labels = np.zeros((N))
            mnist_full._index_in_epoch = 0
            for i in range(all_batches):
                x, y = mnist_full.next_batch(FLAGS.batch_size)
                [xx] = sess.run([mean_x], {input_tensor: x})
                features[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :] = xx
                labels[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size] = y
            model = BayesianGaussianMixture(n_components = FLAGS.T, max_iter = 300, covariance_type = 'diag', weight_concentration_prior=2)
            model.fit(features)
            preds = model.predict(features)
            pis = model.weights_
            mus = model.means_
            precisions = model.precisions_

            print("------------> acc: %f, nmi: %f" % (cluster_acc(preds, labels), cluster_nmi(preds, labels)))
            # print(pis)
            # imgs = sess.run(sampled_tensor)
            # for k in range(FLAGS.batch_size):
            #     imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
            #     if not os.path.exists(imgs_folder):
            #         os.makedirs(imgs_folder)

            #     imsave(os.path.join(imgs_folder, '%d.png') % k,
            #            imgs[k].reshape(28, 28))
