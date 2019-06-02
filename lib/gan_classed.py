from gan_misc import DiscriminatorNetwork_Cond as Discriminator
from gan_misc import GeneratorNetwork_Cond as Generator
'''
init:11Apr18
revamp: Jan2019
Author: OO
Goals:
    - Generalize GAN model to use supplied NNs
    - Using gaussian instead of uniform
    - more general version of (R|B)COA models
Validation:
    - test on mnist data conditional GAN tester

Monitoring:
tensorboard --logdir=worker_1:'./log/train_cgan_worker_1'
'''

import argparse
import sys
import numpy as np
import scipy as sp

_default_reports_ = ['Losses/Generator', 'Losses/Discriminator']

# ML libs
import tensorflow as tf
# import tensorflow.contrib.slim as slim

# GAN auxillary functions
sys.path.append('.')


# Global var defs
__version__ = "0.0.3"
tfboard_path = "./log"
class_name = "cgan_"
learning_rate = 0.0005
_eps_ = 0.1  # 1e-3
gamma_reg = 1e-4  # 0.001
_every_ = 10000
_noiseDim_ = 1


class Conditional_GAN():
    def __init__(self, data,
                 indices_CvG,
                 noiseDim=_noiseDim_,
                 generatorTemplateFxn=Generator,
                 discriminatorTemplateFxn=Discriminator,
                 genScope="generator",
                 genSpec=(32, 256, 64),
                 discScope="discriminator",
                 discSpec=(32, 256, 64),
                 name="worker_1",
                 model_path=tfboard_path,
                 learning_rates=(learning_rate, learning_rate)
                 ):  # input_shape=(7, 7, 2),
        self.name = class_name + str(name)
        self.model_path = model_path+"/train_"+str(self.name)
        self.summary_writer = tf.summary.FileWriter(logdir=self.model_path)  # ./log/train_cgan_worker_1
        self.autosave_every = _every_
        self.report_labels=_default_reports_
        (lrg, lrd) = learning_rates

        # Training exemplar data + dim info
        # self.class_dim, self.cand_dim, self.n_size = input_shape
        self.dataP = data
        self.n_size = noiseDim
        self.classInds, self.genInds = indices_CvG
        self.class_dim = len(indices_CvG[0])
        self.cand_dim = len(indices_CvG[1])
        self.npts, self.inDim = self.dataP.shape  # inDim may not be = gen dim

        # monitors
        self.gen_losses = []
        self.disc_losses = []

        # network params
        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate=lrg)
        self.optimizer_disc = tf.train.AdamOptimizer(learning_rate=lrd)
        self.gamma_reg = gamma_reg

        with tf.variable_scope(self.name):
            # placeholders
            self.n_input = tf.placeholder(tf.float32, shape=[None, self.n_size], name='noise_input')
            self.class_input = tf.placeholder(
                tf.float32, shape=[None, self.class_dim], name='class_input')  # for RCOA
            self.d_input = tf.placeholder(
                tf.float32, shape=[None, self.cand_dim], name='g_to_d_input')

            # Build Generator Net
            with tf.variable_scope('generator') as gen_scope:
                self.gen_sample = generatorTemplateFxn(self.class_input, self.n_input,
                                                       out_dim=self.cand_dim,
                                                       hSeq=genSpec)
            # Build 2 Discriminator Networks (one from noise input, one from generated samples)
            with tf.variable_scope('discriminator') as dis_scope:
                self.disc_real = discriminatorTemplateFxn(self.class_input, self.d_input,
                                                          hSeq=discSpec)  # D(x)
            with tf.variable_scope(dis_scope, reuse=True): # reusing same scope as D(x)
                self.disc_fake = discriminatorTemplateFxn(self.class_input, self.gen_sample,
                                                          hSeq=discSpec)  # D(G(z))

            # Originally:
            self.gen_loss = -tf.reduce_mean(tf.log(self.disc_fake))
            self.disc_loss = -tf.reduce_mean(
                (1-_eps_)*tf.log(self.disc_real) +
                tf.log(1. - self.disc_fake)
            )
            # # Loss functions: using logit NN outputs and one-sided label-smoothing on D(.)
            # self.gen_loss = -tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake,
            #                                             labels=tf.fill(tf.shape(self.disc_fake), 1.))
            # )
            # self.disc_loss = -tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_real,
            #                                             labels=tf.fill(tf.shape(self.disc_real), 1.)) +
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake,
            #                                             labels=tf.fill(tf.shape(self.disc_fake), 0.))
            #     )

            # Isolate G-vs-D variables to optimize
            self.gen_vars = [v for v in tf.trainable_variables() if "generator" in v.name]
            self.disc_vars = [v for v in tf.trainable_variables() if "discriminator" in v.name]

            # Create training operations
            self.train_gen = self.optimizer_gen.minimize(
                self.gen_loss, var_list=self.gen_vars)
            self.train_disc = self.optimizer_disc.minimize(
                self.disc_loss, var_list=self.disc_vars)
        return

    def init_graph(self, sess):
        # initialize variables for the attached session
        with sess.as_default(), sess.graph.as_default():
            sess.run(tf.global_variables_initializer())
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()
        print("Tensorboard logs in: ", self.model_path)
        return

    def work(self, sess, saver, num_epochs, batch_sz=96):
        '''Works through the data for given number of epochs'''
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            # log graph for visualization
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()

            for tk in range(num_epochs):
                # sample cases & Train
                stats = list(self.train(sess, batch_sz))
                self.model_summary(tk, stats, labels_=self.report_labels)
                # save model parameters periodically
                if tk % self.autosave_every == 0:
                    self.model_save(sess, saver,
                                    tk, stats, labels_=self.report_labels)
        return

    def train(self, sess, n):
        '''Updates the GAN networks using gradients from loss;
        Generate network statistics to periodically save.
        Samples a batch (size, n) of +ve exemplars.'''
        inds = np.random.choice(
            len(self.dataP),
            min(n, len(self.dataP)),
            replace=False
        )
        tmp = self.dataP[inds, :]
        # Define input dict
        feed_dict = {
            self.class_input: tmp[:, self.classInds],
            self.d_input: tmp[:, self.genInds],
            # Generate noise to feed to the generator
            self.n_input: np.random.uniform(-1., 1., size=[n, self.n_size])
        }
        # Train
        _, _, gl, dl = sess.run(
            [self.train_gen, self.train_disc,
             self.gen_loss, self.disc_loss],
            feed_dict=feed_dict
        )
        self.gen_losses.append(gl)
        self.disc_losses.append(dl)
        return gl, dl

    def generate(self, sess, condn_classes):
        '''Run generator NN for samples.
        needs: class-conditional indices to generate responses to.'''
        #inds =  condn_classes.flatten()[...,None]
        feed_dict = {
            self.class_input: condn_classes,
            self.n_input: np.random.uniform(-1., 1., size=[len(condn_classes), self.n_size])
        }
        # Generate
        gsamps = sess.run(self.gen_sample, feed_dict=feed_dict)
        return gsamps

    def pretrainD(self, sess, n=50, batch_sz=96):
        '''Pre-train Discriminator on n episodes with 50% fake/true.
        Responding to current version of G(.)
        '''
        data_pre = self.basedf.values
        for _ in range(n):
            inds = np.random.choice(len(data_pre),
                                    size=batch_sz, replace=True)
            tmp = data_pre[inds, :]
            # Define input dict
            feed_dict = {
                self.class_input: tmp[:, self.classInds],
                self.d_input: tmp[:, self.genInds],
                self.n_input: np.random.uniform(-1., 1., size=[batch_sz, self.n_size])
            }
            # Train
            _, _, dl_pre = sess.run(
                [self.gen_loss, self.train_disc, self.disc_loss],
                feed_dict=feed_dict
            )
            self.disc_losses.append(dl_pre)
        return

    def rateSamples(self, sess, condns, resps):
        '''Evaluate the quality of generated responses to given conditions.
        needs:
            - conditional classes tensor (condns)
            - generated responses tensor (resps)
        gives:
            - tensor of discriminator scores (in [0,1]) for given scenarios.
        '''
        feed_dict = {
            self.class_input: condns,
            self.d_input: resps,
            self.n_input: np.random.uniform(-1., 1.,
                                            size=[len(condns), self.n_size])
        }
        # Rate
        disc_ratings = sess.run(self.disc_real, feed_dict=feed_dict)
        # disc_ratings = sess.run(self.disc_fake, feed_dict=feed_dict)
        return disc_ratings

    def model_summary(self, tk, stats_, labels_=_default_reports_):
        summary = tf.Summary()
        for k in range( min(len(stats_), len(labels_) ) ):
            summary.value.add(tag=labels_[k], simple_value=float(stats_[k]))
        self.summary_writer.add_summary(summary, tk)
        self.summary_writer.flush()
        return

    def model_save(self, sess, saver, tk,  stats_, labels_=_default_reports_):
        print('Step {}: Stats({}): ( {} )'.format(tk, labels_, stats_))
        if not any(np.isnan(stats_)):
            chkpt_model = self.model_path + '/model-' + str(tk) + '.cptk'
            self.last_good_model = saver.save(sess, chkpt_model)
            print("Saved Model")
        else:
            print("Model problems... Not saved!")
        return
