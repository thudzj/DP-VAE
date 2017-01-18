from __future__ import absolute_import, division, print_function

import math
import os
import sys
import numpy as np
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from util.dataset import make_dataset
from util.metrics import cluster_acc, cluster_nmi

#from deconv import deconv2d
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
flags.DEFINE_string("logdir", "", "dir for tensorboard logs")
flags.DEFINE_integer("hidden_size", 50, "size of the hidden VAE unit")
flags.DEFINE_integer("T", 10, "level of truncation")
flags.DEFINE_float("lam", 1.0, "weight of the regularizer")
flags.DEFINE_integer("s", 100, "number of samples for testing")
FLAGS = flags.FLAGS

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

def _weight_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.random_normal_initializer(stddev=0.001))

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.0, dtype=tf.float32))

def encoder(input_tensor):
    W_fc1 = _weight_variable('W_1', [784, 500])
    b_fc1 = _bias_variable('b_1', [500])
    W_fc2 = _weight_variable('W_2', [500, FLAGS.hidden_size * 2])
    b_fc2 = _bias_variable('b_2', [FLAGS.hidden_size * 2])
    x_attributes = tf.matmul(tf.nn.relu(tf.matmul(input_tensor, W_fc1) + b_fc1), W_fc2) + b_fc2
    mean_x = x_attributes[:, :FLAGS.hidden_size]
    logcov_x = x_attributes[:, FLAGS.hidden_size:]
    return mean_x, logcov_x

def decoder(mean=None, logcov=None, s=None):
    stddev = tf.sqrt(tf.exp(logcov))
    if s is None:
        epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
        input_sample = mean + epsilon * stddev
    else:
        epsilon = tf.random_normal([s, FLAGS.batch_size, FLAGS.hidden_size])
        # q_x = tf.exp(-0.5 * tf.reduce_sum(tf.square(epsilon), 2))
        input_sample = tf.expand_dims(mean, 0) + epsilon * tf.expand_dims(stddev, 0)
        input_sample = tf.reshape(input_sample, [-1, FLAGS.hidden_size])
    W_fc1 = _weight_variable('W_1', [FLAGS.hidden_size, 500])
    b_fc1 = _bias_variable('b_1', [500])
    W_fc2 = _weight_variable('W_2', [500, 784])
    b_fc2 = _bias_variable('b_2', [784])
    return tf.nn.sigmoid(tf.matmul(tf.nn.relu(tf.matmul(input_sample, W_fc1) + b_fc1), W_fc2) + b_fc2), input_sample, epsilon

def sample(qeta_mu, sigma_px):
    epsilon = tf.random_normal([FLAGS.T, int(FLAGS.batch_size / FLAGS.T), FLAGS.hidden_size])
    means = tf.expand_dims(tf.transpose(qeta_mu), 1)
    covs = tf.expand_dims(tf.transpose(sigma_px), 1)
    input_sample = tf.reshape(epsilon * covs + means, [-1, FLAGS.hidden_size])
    W_fc1 = _weight_variable('W_1', [FLAGS.hidden_size, 500])
    b_fc1 = _bias_variable('b_1', [500])
    W_fc2 = _weight_variable('W_2', [500, 784])
    b_fc2 = _bias_variable('b_2', [784])
    return tf.reshape(tf.nn.sigmoid(tf.matmul(tf.nn.relu(tf.matmul(input_sample, W_fc1) + b_fc1), W_fc2) + b_fc2), [FLAGS.T, int(FLAGS.batch_size / FLAGS.T), 784])

def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
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
    sigma_0 = 100.0 # set a big variance so that the parameters will be learned from data
    return -0.5 * tf.reduce_sum(1 + 2 * tf.log(sigma / sigma_0)
                - tf.square(sigma) / tf.square(sigma_0)
                - tf.square(mu - mu_0) / tf.square(sigma_0))

def get_S_loss_hao(mean_x, logcov_x, qv_alpha, qv_beta, qeta_mu, qeta_sigma, sigma_px, epsilon = 1e-8):
    S1 = tf.digamma(qv_alpha) - tf.digamma(qv_alpha + qv_beta) 
    S2 = tf.cumsum(tf.digamma(qv_beta) - tf.digamma(qv_alpha + qv_beta))

    mean_x_expand = tf.expand_dims(mean_x, 1)
    logcov_x_expand = tf.expand_dims(logcov_x, 1)
    qeta_mu_expand = tf.expand_dims(tf.transpose(qeta_mu), 0)
    qeta_sigma_expand = tf.expand_dims(tf.transpose(qeta_sigma), 0)
    sigma_px_expand = tf.expand_dims(tf.transpose(sigma_px), 0)
    S3 = 0.5 * tf.reduce_sum(1 + logcov_x_expand - 2 * tf.log(sigma_px_expand) \
            - (tf.exp(logcov_x_expand) + tf.square(qeta_sigma_expand) \
            + tf.square(mean_x_expand - qeta_mu_expand)) / tf.square(sigma_px_expand), 2)
    S = S3 + tf.concat(0, [S1, [0.0]]) + tf.concat(0, [[0.0], S2])
    # get the variational distribution q(z)
    S_max = tf.reduce_max(S, reduction_indices=1)
    S_whiten = S - tf.expand_dims(S_max, 1)
    qz = tf.exp(S_whiten) / tf.expand_dims(tf.reduce_sum(tf.exp(S_whiten), 1), 1)
    # Summarize the S loss
    # S_loss = -tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S), 1)))
    S_loss = -tf.reduce_sum(S_max) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(S - tf.expand_dims(S_max, 1)), 1) + epsilon))
    return S_loss, qz, S

def gaussian_mixture_pdf(mu, sigma, x, pi):
    mu_expand = tf.reshape(tf.transpose(mu), [FLAGS.T, 1, 1, FLAGS.hidden_size])
    sigma_expand = tf.reshape(tf.transpose(sigma), [FLAGS.T, 1, 1, FLAGS.hidden_size])
    return tf.reduce_sum(1 / tf.sqrt(tf.reduce_prod(sigma_expand, 3)) 
                        * tf.exp(-0.5 * tf.reduce_sum(tf.square(x - mu_expand) / sigma_expand, 3)) * tf.reshape(pi, [-1, 1, 1]), 0)

def get_marginal_likelihood(yt, mean_yt, xt, s, alpha, beta, eta_mu, eta_sigma, eps, sigma_px, epsilon = 1e-8):
    yt_expand = tf.expand_dims(yt, 0)
    mean_yt = tf.reshape(mean_yt, [s, FLAGS.batch_size, 784])
    xt = tf.reshape(xt, [1, s, FLAGS.batch_size, FLAGS.hidden_size])
    # p_ygivenx = tf.reduce_prod(tf.pow(mean_yt, yt_expand) * tf.pow(1 - mean_yt, 1 - yt_expand), axis=2)
    v = alpha / (alpha + beta)
    pi = tf.concat(0, [v, [1.0]]) * tf.concat(0, [[1.0], tf.cumprod(1 - v)])
    p_x = gaussian_mixture_pdf(eta_mu, tf.square(eta_sigma) + tf.square(sigma_px), xt, pi)
    log_p_y_s = tf.reduce_sum(yt_expand * tf.log(mean_yt + epsilon) \
        + (1.0 - yt_expand) * tf.log(1.0 - mean_yt + epsilon), 2) \
        + tf.log(p_x) \
        + 0.5 * tf.reduce_sum(tf.square(eps), 2)
    log_p_y_s_max = tf.reduce_max(log_p_y_s, reduction_indices=0)
    log_p_y = tf.log(tf.reduce_mean(tf.exp(log_p_y_s - log_p_y_s_max), 0)) + log_p_y_s_max
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

    with tf.variable_scope("encoder") as scope:
        mean_x, logcov_x = encoder(input_tensor)
    with tf.variable_scope("decoder") as scope:
        output_tensor, _, _ = decoder(mean_x, logcov_x)

    with tf.variable_scope("encoder", reuse=True) as scope:
        mean_xt, logcov_xt = encoder(input_tensor)
    with tf.variable_scope("decoder", reuse=True) as scope:
        mean_yt, xt, eps = decoder(mean_xt, logcov_xt, FLAGS.s)

    ''' edit by hao'''
    # first, get the reconstruction term E_q(X|Y) log p(Y|X)
    # which is the cross entory loss between output and input
    rec_loss = 1 / FLAGS.batch_size * get_reconstruction_cost(output_tensor, input_tensor)

    # second, get the q(v|Y) reg loss E_q log \frac{q(v|alpha, beta)}{q(v|Y)}
    qv_alpha = tf.Variable((1 / (FLAGS.T - np.arange(1, FLAGS.T))).astype(np.float32), name = "qv_alpha") # vi parameters
    qv_beta = tf.Variable(tf.ones([FLAGS.T - 1]), name = "qv_beta")  # vi parameters
    # alpha_0 = (1 / (FLAGS.T - np.arange(1, FLAGS.T))).astype(np.float32)
    alpha_0 = np.ones([FLAGS.T - 1]).astype(np.float32)
    beta_0 = 100 * np.ones([FLAGS.T - 1]).astype(np.float32)
    qv_reg_loss = 1 / FLAGS.batch_size * get_qv_reg_loss(qv_alpha, qv_beta, alpha_0, beta_0)

    # third, get the q(\eta|Y) reg loss E_q log \frac{q(\eta|\mu_0, \sigma_0^2)}{q(\eta | Y)}
    qeta_mu = tf.Variable(tf.random_uniform([FLAGS.hidden_size, FLAGS.T]), name = 'qeta_mu') # vi parameters 
    qeta_sigma = tf.Variable(tf.random_uniform([FLAGS.hidden_size, FLAGS.T]), name = 'qeta_sigma') # vi parameters
    qeta_reg_loss = 1 / FLAGS.batch_size * get_qeta_reg_loss(qeta_mu, qeta_sigma)

    # forth, get the remained loss E_q log \frac {p(z|v) p(x|z,y)} {q(z) q(x|y)}
    sigma_px = tf.Variable(tf.ones([FLAGS.hidden_size, FLAGS.T]), name = 'sigma_px', trainable=False)
    S_loss, qz, S = get_S_loss_hao(mean_x, logcov_x, qv_alpha, qv_beta, qeta_mu, qeta_sigma, sigma_px)
    S_loss = 1 / FLAGS.batch_size * S_loss 
    regularizer = - tf.reduce_sum(tf.log(qz + 1e-8) * qz) / FLAGS.batch_size
    qz_all = tf.reduce_mean(qz, axis = 0)
    regularizer_all = tf.reduce_sum(tf.log(qz_all + 1e-8) * qz_all)
    '''END edit by hao'''

    with tf.variable_scope("decoder", reuse=True) as scope:
        samples = sample(qeta_mu, sigma_px)

    vae_loss = rec_loss #+ tf.reduce_sum(0.5 * (tf.square(mean_x) + tf.exp(logcov_x) - logcov_x - 1.0)) / FLAGS.batch_size
    overall_loss = qv_reg_loss + qeta_reg_loss + rec_loss + S_loss + regularizer * 10 + regularizer_all * 100
    log_p_yt = get_marginal_likelihood(input_tensor, mean_yt, xt, FLAGS.s, qv_alpha, qv_beta, qeta_mu, qeta_sigma, eps, sigma_px)
    # Create optimizers
    encoder_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    updates_per_epoch = int(N / FLAGS.batch_size)

    # create the optimizer
    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, updates_per_epoch * 50, 1.0, staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0003, beta1 = 0.95, beta2 = 0.999, epsilon=1.0)
    pre_optimizer = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.95, beta2 = 0.999, epsilon=1.0)
    pretraining_step = pre_optimizer.minimize(vae_loss, var_list = encoder_trainables + decoder_trainables)
    learning_step = optimizer.minimize(overall_loss, var_list = encoder_trainables + decoder_trainables + [qeta_mu, qeta_sigma, qv_alpha, qv_beta], global_step = global_step)

    #global_step_vi = tf.Variable(0, trainable=False)
    #learning_rate_vi = tf.train.exponential_decay(FLAGS.learning_rate, global_step_vi, updates_per_epoch * 50, 0.9, staircase = True)
    #optimizer_vi = tf.train.AdamOptimizer(learning_rate_vi, epsilon=1.0)
    #learning_step_vi = optimizer.minimize(overall_loss, var_list = [qv_alpha, qv_beta], global_step = global_step_vi)

    tf.summary.scalar('overall_loss', overall_loss)
    tf.summary.scalar('rec_loss', rec_loss)
    tf.summary.scalar('S_loss', S_loss)
    tf.summary.scalar('qv_reg_loss', qv_reg_loss)
    tf.summary.scalar('qeta_reg_loss', qeta_reg_loss)
    merged = tf.summary.merge_all()
    loader = tf.train.Saver(encoder_trainables + decoder_trainables + [qv_alpha, qv_beta, qeta_mu, qeta_sigma])
    
    save_dir = 'tb/dp_vae_mnist/' + FLAGS.logdir
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(save_dir, sess.graph)
        sess.run(init)

        print("Pretraining the dp-vae model")
        for iii in range(100):
            ve_loss = 0.0
            mnist_train._index_in_epoch = 0
            for i in range(updates_per_epoch):
                x, _ = mnist_train.next_batch(FLAGS.batch_size)
                _, vae_loss_ = sess.run([pretraining_step, vae_loss], {input_tensor: x}) 
                ve_loss += vae_loss_
            ve_loss /= updates_per_epoch
            print("Pretrain vae loss: %f" % (ve_loss))
            if iii % 10 == 0:
                mnist_train._index_in_epoch = 0
                features = np.zeros([N, FLAGS.hidden_size])
                labels = np.zeros((N))
                for i in range(updates_per_epoch):
                    x, y = mnist_train.next_batch(FLAGS.batch_size)
                    [mean_x_] = sess.run([mean_x], {input_tensor: x})
                    features[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :] = mean_x_
                    labels[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size] = y

                model = BayesianGaussianMixture(n_components = FLAGS.T, max_iter = 300, covariance_type = 'diag', weight_concentration_prior=2)
                model.fit(features)
                preds = model.predict(features)
                print("------------> Fit a Bayesian GMM: acc: %f, nmi: %f" % (cluster_acc(preds, labels), cluster_nmi(preds, labels)))

        # init other variational distributions
        assign_op = qeta_mu.assign(model.means_.T)
        sess.run(assign_op)
        # assign_op = sigma_px.assign(model.covariances_.T)
        # sess.run(assign_op)

        # init the encoder and decoder parameters
        #print("restore the encoder and decoder parameters")
        #loader.restore(sess, "trained/initialization.ckpt")
        print("Training the dp-vae model")
        best_train_rec, best_test_rec, best_val_rec = 999999, 9999999, 9999999
        best_val_heldout, best_test_heldout = 999999, 999999
        mnist_train._index_in_epoch = 0
        for epoch in range(FLAGS.max_epoch):
            # first, let train the encoder and decoder for a while:
            overall_loss_total, rec_loss_total, S_loss_total, qeta_reg_loss_total, qv_reg_loss_total = 0.0, 0.0, 0.0, 0.0, 0.0
            stats = np.zeros([FLAGS.T])
            for i in range(updates_per_epoch):
                x, _ = mnist_train.next_batch(FLAGS.batch_size)
                _, overall_loss_, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_, qz_, summary, step = sess.run([learning_step, overall_loss, rec_loss, S_loss, qeta_reg_loss, qv_reg_loss, qz, merged, global_step], {input_tensor: x}) 
                overall_loss_total += overall_loss_ 
                #print("epoch %d: iter %d, rec_loss: %f, S_loss: %f, qeta_loss: %f, qv_loss: %f" % (epoch, i, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_))
                rec_loss_total += rec_loss_
                S_loss_total += S_loss_
                qeta_reg_loss_total += qeta_reg_loss_
                qv_reg_loss_total += qv_reg_loss_
                train_writer.add_summary(summary, step)
                assignment_ = np.zeros((FLAGS.batch_size, FLAGS.T)) 
                assignment_[np.arange(FLAGS.batch_size), np.argmax(qz_, axis = 1)] = 1
                stats += np.sum(assignment_, axis = 0)
            overall_loss_total = overall_loss_total / updates_per_epoch 
            rec_loss_total = rec_loss_total / updates_per_epoch
            S_loss_total = S_loss_total / updates_per_epoch
            qeta_reg_loss_total = qeta_reg_loss_total / updates_per_epoch
            qv_reg_loss_total = qv_reg_loss_total / updates_per_epoch
            if rec_loss_total <= best_train_rec:
                best_train_rec = rec_loss_total
            print("train ELBO: %f, rec_LL: %f, S_LL: %f, qeta_reg_LL: %f, qv_reg_LL: %f, epoch %d..." 
                % (-overall_loss_total, -rec_loss_total, -S_loss_total, -qeta_reg_loss_total, -qv_reg_loss_total, epoch))
            print("Assigments: ", stats)

            def eval_ELBO(dataset, name):
                dataset._index_in_epoch = 0
                num_iter = int(dataset.num_examples / FLAGS.batch_size)
                overall_loss_total, rec_loss_total, S_loss_total, qeta_reg_loss_total, qv_reg_loss_total, heldout_ll = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for i in range(num_iter):
                    x, _ = dataset.next_batch(FLAGS.batch_size)
                    overall_loss_, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_ , log_p_yt_\
										    = sess.run([overall_loss, rec_loss, S_loss, qeta_reg_loss, qv_reg_loss, log_p_yt], \
						    										{input_tensor: x})                  
                    overall_loss_total += overall_loss_ 
                    #print("epoch %d: iter %d, rec_loss: %f, S_loss: %f, qeta_loss: %f, qv_loss: %f" % (epoch, i, rec_loss_, S_loss_, qeta_reg_loss_, qv_reg_loss_))
                    rec_loss_total += rec_loss_
                    S_loss_total += S_loss_
                    qeta_reg_loss_total += qeta_reg_loss_
                    qv_reg_loss_total += qv_reg_loss_
                    heldout_ll += log_p_yt_
                overall_loss_total = overall_loss_total / num_iter 
                rec_loss_total = rec_loss_total / num_iter
                S_loss_total = S_loss_total / num_iter
                qeta_reg_loss_total = qeta_reg_loss_total / num_iter
                qv_reg_loss_total = qv_reg_loss_total / num_iter
                heldout_ll = heldout_ll / num_iter
                print("%s ELBO: %f, rec_LL: %f, S_LL: %f, qeta_reg_LL: %f, qv_reg_LL: %f, heldout_nll: %f..." 
                    % (name, -overall_loss_total, -rec_loss_total, -S_loss_total, -qeta_reg_loss_total, -qv_reg_loss_total, heldout_ll))
                return rec_loss_total, heldout_ll

            val_rec, val_heldout = eval_ELBO(mnist_val, 'val')
            test_rec, test_heldout = eval_ELBO(mnist_test, 'test')
            if val_rec < best_val_rec:
                best_val_rec = val_rec
            if test_rec < best_test_rec:
                best_test_rec = test_rec
            if -val_heldout < best_val_heldout:
                best_val_heldout = -val_heldout
            if -test_heldout < best_test_heldout:
                best_test_heldout = -test_heldout

            if epoch % 20 == 0:
                [imgs] = sess.run([samples])
                imgs_folder = os.path.join(save_dir, 'imgs', str(epoch))
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)
                for k in range(FLAGS.T):
                    for j in range(int(FLAGS.batch_size / FLAGS.T)):
                        imsave(os.path.join(imgs_folder, '%d_%d.png') % (k, j), imgs[k][j].reshape(28, 28))
            #evaluate the marginal likelihood.
        print("best train rec %f, val rec %f, test rec %f.." % (-best_train_rec, -best_val_rec, -best_test_rec))
        print("best val heldout-ll %f, test heldout-ll %f.." % (-best_val_heldout, -best_test_heldout))
        loader.save(sess, train_writer.get_logdir() + '/final_model.ckpt')
    train_writer.close()
