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
from sklearn.mixture import GaussianMixture

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("batch_size", 200, "batch size") #128
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch") #1000
flags.DEFINE_integer("max_epoch", 300, "max epoch") #100
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_integer("T", 10, "level of truncation")
flags.DEFINE_float("alpha_0", 1.0, "alpha_0 for the prior Beta distribution")
flags.DEFINE_float("beta_0", 1.0, "beta_0 for the prior Beta distribution")
flags.DEFINE_float("sigma2", 1.0, "covariance for the mixtured gaussian components")
flags.DEFINE_integer("rnncell_size", 1000, "the hidden size of rnn cell")
FLAGS = flags.FLAGS

def encoder(input_tensor, state_tensor):
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
    mean_x = x_attributes[:, :FLAGS.hidden_size]
    logcov_x = x_attributes[:, FLAGS.hidden_size:]

    # A = (pt.template('x').gru_cell(num_units=FLAGS.rnncell_size, state=pt.UnboundVariable('state')))
    # for i in range(FLAGS.batch_size):
        # if i == 0:
            # lstm_feature, [new_state] = A.construct(x=tf.expand_dims(mid_features.tensor[i], 0), state=state_tensor)
        # else:
            # lstm_feature, [new_state] = A.construct(x=tf.expand_dims(mid_features.tensor[i], 0), state=new_state)
    # clusters_attributes = (pt.wrap(lstm_feature).fully_connected(FLAGS.T*(2*FLAGS.hidden_size+2)-2, activation_fn=None)).tensor

    lstm_features, [new_state] = mid_features.gru_cell(num_units=FLAGS.rnncell_size, state=state_tensor)
    clusters_attributes = (pt.wrap(tf.reduce_mean(lstm_features, 0, keep_dims=True)).fully_connected(FLAGS.T*(2*FLAGS.hidden_size+2)-2, activation_fn=None)).tensor

    mean_eta = tf.reshape(clusters_attributes[:, :FLAGS.T*FLAGS.hidden_size], [FLAGS.T, FLAGS.hidden_size])
    logcov_eta = tf.reshape(clusters_attributes[:, FLAGS.T*FLAGS.hidden_size: 2*FLAGS.T*FLAGS.hidden_size], [FLAGS.T, FLAGS.hidden_size])
    alpha = tf.exp(tf.squeeze(clusters_attributes[:, 2*FLAGS.T*FLAGS.hidden_size:2*FLAGS.T*FLAGS.hidden_size+FLAGS.T-1]))
    beta = tf.exp(tf.squeeze(clusters_attributes[:, 2*FLAGS.T*FLAGS.hidden_size+FLAGS.T-1:]))

    # list_attributes = []
    # for i in range(FLAGS.T):
    #     list_attributes.append(tf.reduce_mean((mid_features.fully_connected(FLAGS.hidden_size * 2 + 2, activation_fn=None)).tensor, 0))
    # clusters_attributes = tf.pack(list_attributes)

    # mean_eta = clusters_attributes[:, :FLAGS.hidden_size]
    # logcov_eta = clusters_attributes[:, FLAGS.hidden_size:FLAGS.hidden_size*2]
    # alpha = tf.squeeze(tf.exp(clusters_attributes[:-1, FLAGS.hidden_size*2]))
    # beta = tf.squeeze(tf.exp(clusters_attributes[:-1, FLAGS.hidden_size*2 + 1]))

    return mean_x, logcov_x, mean_eta, logcov_eta, alpha, beta, new_state


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
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()
            # fully_connected(2000, name='decoder_l1', activation_fn=tf.nn.relu).
            # fully_connected(500, name='decoder_l2', activation_fn=tf.nn.relu).
            # fully_connected(500, name='decoder_l3', activation_fn=tf.nn.relu).
            # fully_connected(784, name='decoder_l4', activation_fn=tf.nn.sigmoid)
            ).tensor


def kl_Gaussian(mean, logcov, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean: 
        logcov:
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
    S1 = tf.digamma(alpha) - tf.digamma(alpha + beta)
    S2 = tf.cumsum(tf.digamma(beta) - tf.digamma(alpha + beta))
    S = 0.5 * tf.reduce_sum( \
            1 + logcov_x_pad - math.log(sigma2) \
            - (tf.exp(logcov_x_pad) + tf.exp(logcov_eta_pad) + tf.square(mean_x_pad - mean_eta_pad)) / sigma2 , 2 \
        ) \
        + tf.concat(0, [S1, tf.constant([0.0])]) + tf.concat(0, [tf.constant([0.0]), S2])

    assignments = tf.argmax(S, dimension=1)
    S_max = tf.reduce_max(S, reduction_indices=1)
    S_loss = -tf.reduce_sum(S_max) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S - tf.expand_dims(S_max, 1)), reduction_indices = 1) + epsilon))
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
def get_regularization(mean_eta, logcov_eta):
    #reg = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(mean_eta_tensor - mean_eta), 1)))
    reg = -tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_mean(mean_eta, 0) - mean_eta), 1)))
    #reg = -tf.reduce_sum(tf.square(tf.reduce_mean(mean_eta, 0) - mean_eta))
    # cov_eta = tf.exp(logcov_eta)
    # reg += tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_mean(cov_eta, 0) - cov_eta), 1)))
    return reg*1e-6

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
    state_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.rnncell_size])
    #mean_eta_tensor = tf.placeholder(tf.float32, [FLAGS.T, FLAGS.hidden_size])

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                mean_x, logcov_x, mean_eta, logcov_eta, alpha, beta, new_state = encoder(input_tensor, state_tensor)
            with tf.variable_scope("decoder") as scope:
                output_tensor = decoder(mean_x, logcov_x)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("decoder", reuse=True) as scope:
                sampled_tensor = decoder()

    kl_eta = kl_Gaussian(mean_eta, logcov_eta)
    kl_alphabeta = kl_Beta(alpha, beta, FLAGS.alpha_0, FLAGS.beta_0)
    assignments, S_loss = get_S_loss(alpha, beta, mean_x, logcov_x, mean_eta, logcov_eta, FLAGS.sigma2)
    rec_loss = get_reconstruction_cost(output_tensor, input_tensor)
    reg = get_regularization(mean_eta, logcov_eta)

    loss = (kl_eta + kl_alphabeta) / float(N) * float(FLAGS.batch_size) + rec_loss + S_loss + reg
    vae_loss = rec_loss + kl_Gaussian(mean_x, logcov_x)
    #loss =  rec_loss + S_loss + reg

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = []
    for grad, var in grads_and_vars:
        if grad is not None:
            clipped_grads_and_vars.append((tf.clip_by_value(grad, -1., 1.), var))
        #print(grad, var)
    train = optimizer.apply_gradients(clipped_grads_and_vars)

    train_vae = pt.apply_optimizer(optimizer, losses=[vae_loss])
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        FLAGS.updates_per_epoch = int(N / FLAGS.batch_size)
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
            loss4 = 0.0
            loss5 = 0.0
            cur_state = np.zeros((FLAGS.batch_size, FLAGS.rnncell_size))
            labels_pred = np.zeros(N)
            labels = np.zeros(N)

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, y = mnist_full.next_batch(FLAGS.batch_size)
                # [m_x] = sess.run([mean_x], {input_tensor: x})
                # g_means = gmm.fit(m_x).means_

                _, cur_state, y_p, loss_value, loss1_, loss2_, loss3_, loss4_, loss5_ = sess.run([train, new_state, assignments, loss, kl_alphabeta, kl_eta, S_loss, rec_loss, reg], {input_tensor: x, state_tensor:cur_state})
                #print(loss1_, loss2_, loss3_, loss4_, loss5_)
                labels[i * FLAGS.batch_size : (i + 1) * FLAGS.batch_size] = y
                labels_pred[i * FLAGS.batch_size : (i + 1) * FLAGS.batch_size] = y_p
                training_loss += loss_value
                loss1 += loss1_
                loss2 += loss2_
                loss3 += loss3_
                loss4 += loss4_
                loss5 += loss5_

            training_loss = training_loss / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss1 = loss1 / FLAGS.updates_per_epoch / float(N)
            loss2 = loss2 / FLAGS.updates_per_epoch / float(N)
            loss3 = loss3 / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss4 = loss4 / FLAGS.updates_per_epoch / FLAGS.batch_size
            loss5 = loss5 / FLAGS.updates_per_epoch / FLAGS.batch_size
                
            print("Loss: %f, kl_alphabeta: %f, kl_eta: %f, S_loss: %f, rec_loss: %f, reg: %f, acc: %f, nmi: %f" 
                % (training_loss, loss1, loss2, loss3, loss4, loss5, cluster_acc(labels_pred, labels), cluster_nmi(labels_pred, labels)))

            # imgs = sess.run(sampled_tensor)
            # for k in range(FLAGS.batch_size):
            #     imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
            #     if not os.path.exists(imgs_folder):
            #         os.makedirs(imgs_folder)

            #     imsave(os.path.join(imgs_folder, '%d.png') % k,
            #            imgs[k].reshape(28, 28))
