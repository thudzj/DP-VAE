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
flags.DEFINE_integer("max_epoch", 300, "max epoch") #100
flags.DEFINE_float("learning_rate", 0.003, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 1000, "size of the hidden VAE unit")
flags.DEFINE_integer("T", 10, "level of truncation")
flags.DEFINE_float("lam", 1.0, "weight of the regularizer")

flags.DEFINE_float("alpha_0", 1.0, "alpha_0 for the prior Beta distribution")
flags.DEFINE_float("beta_0", 1.0, "beta_0 for the prior Beta distribution")
FLAGS = flags.FLAGS

def _weight_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.01))

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0, dtype=tf.float32))

def conv_3d(input, in_filters, out_filters, name, ksize = [5, 5, 5], strides = [1, 1, 1], padding='SAME', activation_fn=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        kernel = _weight_variable('weights', ksize + [in_filters, out_filters])
        conv = tf.nn.conv3d(input, kernel, [1] + strides + [1], padding=padding)
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        return activation_fn(bias, name=scope.name)

def deconv_3d(input):
    return input

def pool_3d(input, ksize = [3, 3, 3], strides = [2, 2, 2], padding='SAME'):
    return tf.nn.max_pool3d(input, ksize=[1]+ksize+[1], strides=[1]+strides+[1], padding=padding)

def fc(input, in_filters, out_filters, name, activation_fn=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        weights = _weight_variable('weights', [in_filters, out_filters])
        biases = _bias_variable('biases', [out_filters])
        if not activation_fn is None:
            biases = activation_fn(tf.matmul(input, weights) + biases, name=scope.name)
        return biases

def encoder(input_tensor):
    conv = conv_3d(input_tensor, 3, 16, 'conv1')
    pool = pool_3d(conv)
    conv = conv_3d(pool, 16, 32, 'conv2')
    pool = pool_3d(conv)
    conv = conv_3d(pool, 32, 64, 'conv3_1')
    conv = conv_3d(conv, 64, 64, 'conv3_2')
    conv = conv_3d(conv, 64, 32, 'conv3_3')
    pool = pool_3d(conv)
    dim = np.prod(pool.get_shape().as_list()[1:])
    print dim
    flat = tf.reshape(prev_layer, [-1, dim])
    x_attributes = fc(mid_features, dim, FLAGS.hidden_size * 2, activation_fn=None)
    mean_x = x_attributes[:, :FLAGS.hidden_size]
    logcov_x = x_attributes[:, FLAGS.hidden_size:]
    return mean_x, logcov_x

def decoder(mean=None, logcov=None):
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
            fully_connected(2000, name='decoder_fc1', activation_fn=tf.nn.relu).
            fully_connected(500, name='decoder_fc2', activation_fn=tf.nn.relu).
            fully_connected(500, name='decoder_fc3', activation_fn=tf.nn.relu).
            fully_connected(784, name='decoder_fc4', activation_fn=tf.nn.sigmoid)
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

def kl_Beta(alpha, beta, alpha_0, beta_0):
    return tf.reduce_sum(tf.lgamma(alpha_0) + tf.lgamma(beta_0) - tf.lgamma(alpha_0+beta_0)
                + tf.lgamma(alpha + beta) - tf.lgamma(alpha) - tf.lgamma(beta)
                + (alpha - alpha_0) * tf.digamma(alpha) + (beta - beta_0) * tf.digamma(beta)
                - (alpha + beta - alpha_0 - beta_0) * tf.digamma(alpha + beta))

def get_qv_reg_loss(alpha, beta, alpha_0, beta_0):
    # get the q(v|Y) loss E_q log \frac{q(v|alpha, beta)}{q(v|Y)}
    return kl_Beta(alpha, beta, alpha_0, beta_0)

def get_qeta_reg_loss(mu, sigma):
    # get the q(\eta|Y) reg loss E_q log \frac{q(\eta|\mu_0, \sigma_0^2)}{q(\eta | Y)}
    mu_0 = tf.zeros([FLAGS.hidden_size, 1])  #parameters of p(\eta) ~ N(mu_0, sigma_0^2 I)
    sigma_0 = 100.0 # set a big variance so that the data will be learned from data
    return -0.5 * tf.reduce_sum(1 + 2 * tf.log(sigma / sigma_0)
                - tf.square(sigma) / tf.square(sigma_0)
                - tf.square(mu - mu_0) / tf.square(sigma_0))

def get_S_loss_hao(mean_x, logcov_x, qv_alpha, qv_beta, qeta_mu, qeta_sigma, epsilon = 1e-8):
    sigma_px = 1.0  
    S1 = tf.digamma(qv_alpha) - tf.digamma(qv_alpha + qv_beta) 
    S2 = tf.cumsum(tf.digamma(qv_beta) - tf.digamma(qv_alpha + qv_beta))

    mean_x_expand = tf.expand_dims(mean_x, 1)
    logcov_x_expand = tf.expand_dims(logcov_x, 1)
    qeta_mu_expand = tf.expand_dims(tf.transpose(qeta_mu), 0)
    qeta_sigma_expand = tf.expand_dims(tf.transpose(qeta_sigma), 0)
    S3 = 0.5 * tf.reduce_sum(1 + logcov_x_expand - 2 * tf.log(sigma_px) \
            - (tf.exp(logcov_x_expand) + tf.square(qeta_sigma_expand) \
            + tf.square(mean_x_expand - qeta_mu_expand)) / tf.square(sigma_px), 2)
    S = S3 + tf.concat(0, [S1, [0.0]]) + tf.concat(0, [[0.0], S2])
    # get the variational distribution q(z)
    S_max = tf.reduce_max(S, reduction_indices=1)
    S_whiten = S - tf.expand_dims(S_max, 1)
    qz = tf.exp(S_whiten) / tf.expand_dims(tf.reduce_sum(tf.exp(S_whiten), 1), 1)
    # Summarize the S loss
    # S_loss = -tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S), 1)))
    S_loss = -tf.reduce_sum(S_max) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S - tf.expand_dims(S_max, 1)), 1) + epsilon))
    return S_loss, qz, S

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

    with pt.defaults_scope(activation_fn=tf.nn.relu,
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

    ''' edit by hao'''
    # first, get the reconstruction term E_q(X|Y) log p(Y|X)
    # which is the cross entory loss between output and input
    rec_loss = 1 / FLAGS.batch_size * get_reconstruction_cost(output_tensor, input_tensor)

    # second, get the q(v|Y) reg loss E_q log \frac{q(v|alpha, beta)}{q(v|Y)}
    qv_alpha = tf.Variable(tf.ones([FLAGS.T - 1]), name = "qv_alpha") # vi parameters
    qv_beta = tf.Variable(tf.ones([FLAGS.T - 1]), name = "qv_beta")  # vi parameters
    qv_reg_loss = 1 / FLAGS.batch_size * get_qv_reg_loss(qv_alpha, qv_beta, FLAGS.alpha_0, FLAGS.beta_0)

    # third, get the q(\eta|Y) reg loss E_q log \frac{q(\eta|\mu_0, \sigma_0^2)}{q(\eta | Y)}
    qeta_mu = tf.Variable(tf.random_uniform([FLAGS.hidden_size, FLAGS.T]), name = 'qeta_mu') # vi parameters 
    qeta_sigma = tf.Variable(tf.random_uniform([FLAGS.hidden_size, FLAGS.T]), name = 'qeta_sigma') # vi parameters
    qeta_reg_loss = 1 / FLAGS.batch_size * get_qeta_reg_loss(qeta_mu, qeta_sigma)

    # forth, get the remained loss E_q log \frac {p(z|v) p(x|z,y)} {q(z) q(x|y)}
    S_loss, qz, S = get_S_loss_hao(mean_x, logcov_x, qv_alpha, qv_beta, qeta_mu, qeta_sigma)
    S_loss = 1 / FLAGS.batch_size * S_loss 
    '''END edit by hao'''

    overall_loss = 0.0 * qv_reg_loss + 0.0 * qeta_reg_loss + rec_loss + S_loss
    # Create optimizers
    encoder_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    #nn_grads_and_vars = optimizer.compute_gradients(overall_loss, encoder_trainables + decoder_trainables)
    #nn_grads_and_vars = optimizer.compute_gradients(S_loss + rec_loss, encoder_trainables + decoder_trainables)
    #clipped_grads_and_vars = []
    #for grad, var in grads_and_vars:
      # if grad is not None:
          # clipped_grads_and_vars.append((tf.clip_by_value(grad, -1., 1.), var))
      # #print(grad, var)
    #train_nn = optimizer.apply_gradients(nn_grads_and_vars)
    #dp_grads_and_vars = optimizer.compute_gradients(overall_loss, {qv_alpha, qv_beta, qeta_mu, qeta_sigma})
    #dp_grads_and_vars = optimizer.compute_gradients(S_loss + rec_loss, {qv_alpha, qv_beta, qeta_mu, qeta_sigma})
    #train_dp = optimizer.apply_gradients(dp_grads_and_vars)

    #pretrain_grads_and_vars = optimizer.compute_gradients(vae_loss, encoder_trainables + decoder_trainables)
    #pretrain = optimizer.apply_gradients(pretrain_grads_and_vars)

    #train = pt.apply_optimizer(optimizer, losses=[overall_loss])
    #train_reg = pt.apply_optimizer(optimizer, losses=[loss + reg])
    #train_vae = pt.apply_optimizer(optimizer, losses=[vae_loss])
    #train_rec = pt.apply_optimizer(optimizer, losses=[rec_loss])
    #train_x = pt.apply_optimizer(optimizer, losses=[rec_loss + S_loss])
    #train_model = pt.apply_optimizer(optimizer, losses=[S_loss + qv_reg_loss + qeta_reg_loss])

    updates_per_epoch = int(N / FLAGS.batch_size)

    # create the optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, updates_per_epoch * 50, 0.9, staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
    learning_step = optimizer.minimize(overall_loss, var_list = encoder_trainables + decoder_trainables + [], global_step = global_step)

    global_step_vi = tf.Variable(0, trainable=False)
    learning_rate_vi = tf.train.exponential_decay(FLAGS.learning_rate, global_step_vi, updates_per_epoch * 50, 0.9, staircase = True)
    optimizer_vi = tf.train.AdamOptimizer(learning_rate_vi, epsilon=1.0)
    learning_step_vi = optimizer.minimize(overall_loss, var_list = [qeta_mu, qeta_sigma, qv_alpha, qv_beta], global_step = global_step_vi)

    loader = tf.train.Saver(encoder_trainables + decoder_trainables)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        '''
        # init the encoder and decoder parameters
        print("restore the encoder and decoder parameters")
        loader.restore(sess, "trained/initialization.ckpt")
          # test the intialization
        features = np.zeros((N, FLAGS.hidden_size))
        labels = np.zeros((N))
        mnist_full._index_in_epoch = 0
        for i in range(updates_per_epoch):
            x, y = mnist_full.next_batch(FLAGS.batch_size)
            [xx] = sess.run([mean_x], {input_tensor: x})
            features[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :] = xx
            labels[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size] = y
        #model = GaussianMixture(n_components = FLAGS.T, max_iter = 300, covariance_type = 'diag')
        #model.fit(features)
        #preds = model.predict(features)
        #print("---------------------> Fit a GMM: acc: %f, nmi: %f" % (cluster_acc(preds, labels), cluster_nmi(preds, labels)))
        model = BayesianGaussianMixture(n_components = FLAGS.T, max_iter = 300, covariance_type = 'diag', 
                               weight_concentration_prior=2)
        model.fit(features)
        preds = model.predict(features)
        print("------------> Fit a Bayesian GMM: acc: %f, nmi: %f" % (cluster_acc(preds, labels), cluster_nmi(preds, labels)))
        # init other variational distributions
        assign_op = qeta_mu.assign(model.means_.T)
        sess.run(assign_op)
        #assign_op = qeta_sigma.assign(np.sqrt(model.covariances_.T))
        #sess.run(assign_op)
                #qv_alpha = 
                #qv_beta = 
        '''

        print("Training the dp-vae model")
        mnist_full._index_in_epoch = 0
        for epoch in range(FLAGS.max_epoch):
            # first, let train the encoder and decoder for a while:
            training_loss, loss1, loss2, loss3, loss4 = 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(updates_per_epoch):
                x, _ = mnist_full.next_batch(FLAGS.batch_size)
                _, overall_loss_, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_, qv_alpha_, qv_beta_, qeta_mu_, qeta_sigma_, qz_, S_, lr_ \
                    = sess.run([learning_step, overall_loss, rec_loss, S_loss, qeta_reg_loss, qv_reg_loss, \
                                qv_alpha, qv_beta, qeta_mu, qeta_sigma, qz, S, learning_rate], {input_tensor: x})                  
                training_loss += overall_loss_ 
                #print("epoch %d: iter %d, rec_loss: %f, S_loss: %f, qeta_loss: %f, qv_loss: %f" % (epoch, i, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_))
                loss1 += rec_loss_
                loss2 += S_loss_
                loss3 += qeta_reg_loss_
                loss4 += qv_reg_loss_
            training_loss = training_loss / updates_per_epoch 
            loss1 = loss1 / updates_per_epoch
            loss2 = loss2 / updates_per_epoch
            loss3 = loss3 / updates_per_epoch
            loss4 = loss4 / updates_per_epoch
            print("Epoch %d, lr %f, Overall Loss: %f, rec_loss: %f, S_loss: %f, qeta_reg_loss: %f, qv_reg_loss_: %f" 
                % (epoch, lr_, training_loss, loss1, loss2, loss3, loss4))

            # then, train the dp parameters for a while:
            if True:
                training_loss, loss1, loss2, loss3, loss4 = 0.0, 0.0, 0.0, 0.0, 0.0
                for i in range(updates_per_epoch):
                    #pbar.update(i)
                    x, _ = mnist_full.next_batch(FLAGS.batch_size)
                    _, overall_loss_, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_, qv_alpha_, qv_beta_, qeta_mu_, qeta_sigma_, qz_, S_, lr_vi_ \
                        = sess.run([learning_step_vi, overall_loss, rec_loss, S_loss, qeta_reg_loss, qv_reg_loss, \
                                    qv_alpha, qv_beta, qeta_mu, qeta_sigma, qz, S, learning_rate_vi], {input_tensor: x})                  
                    training_loss += overall_loss_ 
                    #print("epoch %d: iter %d, rec_loss: %f, S_loss: %f, qeta_loss: %f, qv_loss: %f" % (epoch, i, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_))
                    loss1 += rec_loss_
                    loss2 += S_loss_
                    loss3 += qeta_reg_loss_
                    loss4 += qv_reg_loss_
                training_loss = training_loss / updates_per_epoch
                loss1 = loss1 / updates_per_epoch
                loss2 = loss2 / updates_per_epoch
                loss3 = loss3 / updates_per_epoch
                loss4 = loss4 / updates_per_epoch
                print("Epoch %d, lr %f, Overall Loss: %f, rec_loss: %f, S_loss: %f, qeta_reg_loss: %f, qv_reg_loss_: %f" 
                    % (epoch, lr_vi_, training_loss, loss1, loss2, loss3, loss4))
      

            # evaluation
            features = np.zeros((N, FLAGS.hidden_size))
            labels = np.zeros((N))
            qz_output = np.zeros([N, FLAGS.T])
            mnist_full._index_in_epoch = 0
            for i in range(updates_per_epoch):
                x, y = mnist_full.next_batch(FLAGS.batch_size)
                [xx, qz_batch] = sess.run([mean_x, qz], {input_tensor: x})
                features[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :] = xx
                labels[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size] = y
                qz_output[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :] =  qz_batch
            model = BayesianGaussianMixture(n_components = FLAGS.T, max_iter = 300, covariance_type = 'diag', weight_concentration_prior=2)
            model.fit(features)
            preds = model.predict(features)
            assignments = np.argmax(qz_output, axis = 1)
            print(np.unique(assignments))
            print("-------------------> Fit a Bayesian GMM: acc: %f, nmi: %f" % (cluster_acc(preds, labels), cluster_nmi(preds, labels)))
            print("------------> variational distributions: acc: %f, nmi: %f" % (cluster_acc(assignments, labels), cluster_nmi(assignments, labels)))
            # print(pis)
            # imgs = sess.run(sampled_tensor)
            # for k in range(FLAGS.batch_size):
            #     imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
            #     if not os.path.exists(imgs_folder):
            #         os.makedirs(imgs_folder)

            #     imsave(os.path.join(imgs_folder, '%d.png') % k,
            #            imgs[k].reshape(28, 28))
