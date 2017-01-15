from __future__ import absolute_import, division, print_function

import math
import os
import sys
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
import math
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("batch_size", 100, "batch size") #128
flags.DEFINE_integer("updates_per_epoch", 600, "number of updates per epoch") #1000
flags.DEFINE_integer("max_epoch", 1000, "max epoch") #100
flags.DEFINE_float("learning_rate", 0.01, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("log_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_integer("s", 100, "number of samples for testing")
FLAGS = flags.FLAGS

# tanh 
# hidden size: 500 for mnist
# dim of latent variable: 

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def encoder(input_tensor):
    mid_features = (pt.wrap(input_tensor).
            fully_connected(500, name='encoder_fc1', activation_fn=tf.nn.tanh))
    x_attributes = (mid_features.fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor
    mean_x = x_attributes[:, :FLAGS.hidden_size]
    logcov_x = x_attributes[:, FLAGS.hidden_size:]
    return mean_x, logcov_x

def decoder(mean=None, logcov=None, s=None):
    if mean is None and logcov is None:
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        mean = None
        logcov = None
        stddev = None
        input_sample = epsilon
    else:
        stddev = tf.sqrt(tf.exp(logcov))
        if s is None:
            epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
            input_sample = mean + epsilon * stddev
        else:
            epsilon = tf.random_normal([s, FLAGS.batch_size, FLAGS.hidden_size])
            # q_x = tf.exp(-0.5 * tf.reduce_sum(tf.square(epsilon), 2))
            input_sample = tf.expand_dims(mean, 0) + epsilon * tf.expand_dims(stddev, 0)
            input_sample = tf.reshape(input_sample, [-1, FLAGS.hidden_size])
    return (pt.wrap(input_sample).
            fully_connected(500, name='decoder_fc1', activation_fn=tf.nn.tanh).
            fully_connected(784, name='decoder_fc2', activation_fn=tf.nn.sigmoid)
            ).tensor, input_sample, epsilon

def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                         (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

def get_marginal_likelihood(yt, mean_yt, xt, s, eps, epsilon = 1e-8):
    yt_expand = tf.expand_dims(yt, 0)
    mean_yt = tf.reshape(mean_yt, [s, FLAGS.batch_size, 784])
    xt = tf.reshape(xt, [s, FLAGS.batch_size, FLAGS.hidden_size])
    log_p_y_s = tf.reduce_sum(yt_expand * tf.log(mean_yt + epsilon) \
        + (1.0 - yt_expand) * tf.log(1.0 - mean_yt + epsilon), 2) \
        - 0.5 * tf.reduce_sum(tf.square(xt), 2) \
        + 0.5 * tf.reduce_sum(tf.square(eps), 2)
    
    log_p_y = tf.log(tf.reduce_mean(tf.exp(log_p_y_s), 0))
    return tf.reduce_mean(log_p_y)

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=False)
    train, val, test = mnist.train.images, mnist.validation.images, mnist.test.images 
    train_labels, val_labels, test_labels = mnist.train.labels, mnist.validation.labels, mnist.test.labels
    mnist_train = make_dataset(train, train_labels)
    mnist_val = make_dataset(val, val_labels)
    mnist_test = make_dataset(test, test_labels)

    N = mnist_train.num_examples

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])

    with pt.defaults_scope(activation_fn=tf.nn.tanh,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                mean_x, logcov_x = encoder(input_tensor)
            with tf.variable_scope("decoder") as scope:
                output_tensor, _, _ = decoder(mean_x, logcov_x)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("encoder", reuse=True) as scope:
                mean_xt, logcov_xt = encoder(input_tensor)
            with tf.variable_scope("decoder", reuse=True) as scope:
                mean_yt, xt, eps = decoder(mean_xt, logcov_xt, FLAGS.s)

    ''' edit by hao'''
    # first, get the reconstruction term E_q(X|Y) log p(Y|X)
    # which is the cross entory loss between output and input
    rec_loss = 1 / FLAGS.batch_size * get_reconstruction_cost(output_tensor, input_tensor)
    reg_loss = 1 / FLAGS.batch_size * tf.reduce_sum(0.5 * (tf.square(mean_x) + tf.exp(logcov_x) - logcov_x - 1.0))
    vae_loss = rec_loss + reg_loss
    log_p_yt = get_marginal_likelihood(input_tensor, mean_yt, xt, FLAGS.s, eps)
    

    # Create optimizers
    encoder_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    updates_per_epoch = int(N / FLAGS.batch_size)

    # create the optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, updates_per_epoch * 50, 1.0, staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
    learning_step = optimizer.minimize(vae_loss, var_list = encoder_trainables + decoder_trainables, global_step = global_step)

    # add variable summaries
    tf.summary.scalar('vae_loss', vae_loss)
    tf.summary.scalar('rec_loss', rec_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    merged = tf.summary.merge_all()
		#val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
		#test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    loader = tf.train.Saver(encoder_trainables + decoder_trainables)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(FLAGS.working_directory + 'tb/vae_mnist/train3', sess.graph)
        print("Training the vae model")
        mnist_train._index_in_epoch = 0
        for epoch in range(FLAGS.max_epoch):
            # first, let train the encoder and decoder for a while:
            overall_loss_total, rec_loss_total, reg_loss_total  = 0.0, 0.0, 0.0
            for i in range(updates_per_epoch):
                x, _ = mnist_train.next_batch(FLAGS.batch_size)
                _, overall_loss_, rec_loss_, reg_loss_, lr_, summary, step \
                    = sess.run([learning_step, vae_loss, rec_loss, reg_loss, learning_rate, merged, global_step], {input_tensor: x})                  
                overall_loss_total += overall_loss_ 
                rec_loss_total += rec_loss_
                reg_loss_total += reg_loss_
                train_writer.add_summary(summary, step)
            #print("epoch %d: iter %d, rec_loss: %f, S_loss: %f, qeta_loss: %f, qv_loss: %f" % (epoch, i, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_))
            overall_loss_total = overall_loss_total / updates_per_epoch 
            rec_loss_total = rec_loss_total / updates_per_epoch
            reg_loss_total = reg_loss_total / updates_per_epoch
            print("training ELBO: %f, rec_LL: %f, reg_LL: %f; Epoch %d, lr %f.." % (-overall_loss_total, -rec_loss_total, -reg_loss_total, epoch, lr_))

            def eval_ELBO(dataset, name):
                dataset._index_in_epoch = 0
                num_iter = int(dataset.num_examples / FLAGS.batch_size)
                overall_loss_total, rec_loss_total, reg_loss_total, heldout_ll = 0.0, 0.0, 0.0, 0.0
                for i in range(num_iter):
                    x, _ = dataset.next_batch(FLAGS.batch_size)
                    overall_loss_, rec_loss_, reg_loss_, heldout_ll_ = sess.run([vae_loss, rec_loss, reg_loss, log_p_yt], {input_tensor: x})
                    overall_loss_total += overall_loss_ 
                    rec_loss_total += rec_loss_
                    reg_loss_total += reg_loss_
                    heldout_ll += heldout_ll_
                overall_loss_total = overall_loss_total / num_iter 
                rec_loss_total = rec_loss_total / num_iter 
                reg_loss_total = reg_loss_total / num_iter 
                heldout_ll = heldout_ll / num_iter
                print("%s ELBO: %f, rec_LL: %f, reg_LL: %f, heldout_nll: %f"  
                    % (name, -overall_loss_total, -rec_loss_total, -reg_loss_total, -heldout_ll))

            # evaluate the validation/test ELBO
            eval_ELBO(mnist_val, 'val')
            eval_ELBO(mnist_test, 'test')
            #evaluate the marginal likelihood.
    train_writer.close()
