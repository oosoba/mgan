from gan_misc import DiscriminatorNetwork
from gan_misc import GeneratorNetwork
'''
init:11Apr18
upd: Jan2019
Author: OO
Goals:
    - Generalize GAN model to use supplied NNs (encapsulated in gan_misc)
    - Using gaussian instead of uniform
Monitoring:
tensorboard --logdir=gan_worker_1:'./log/train_gan_worker_1'
'''

# from __future__ import division, absolute_import  # , print_function
import argparse
import sys
import os
import re
import json
import csv
import numpy as np
import scipy as sp

# ML libs
import tensorflow as tf
#import tensorflow.contrib.slim as slim
#import tensorflow.contrib.gan as tfgan
#layers = tf.contrib.layers
#ds = tf.contrib.distributions


# GAN auxillary functions
sys.path.append('.')

# Global var defs
__version__ = "0.0.3"
tfboard_path = "./log"
class_name = "gan_"
learning_rate = 0.0005
gamma_reg = 1e-4  # 0.001
_every_ = 10000
_noiseDim_ = 1


class GAN():
    def __init__(self, data,
                 noiseDim=_noiseDim_,
                 generatorTemplateFxn=GeneratorNetwork,
                 discriminatorTemplateFxn=DiscriminatorNetwork,
                 genScope="generator",
                 genSpec=(32, 256, 64),
                 discScope="discriminator",
                 discSpec=(32, 256, 64),
                 name="worker_1",
                 model_path=tfboard_path,
                 learning_rates=(1e-3, 1e-4)
                 ):
        self.name = "gan_" + str(name)
        self.model_path = model_path+"/train_"+str(self.name)
        self.summary_writer = tf.summary.FileWriter(
            logdir=self.model_path
        )
        self.autosave_every = _every_

        # Training exemplar data + dim info
        self.dataP = data
        self.npts, self.inDim = self.dataP.shape
        self.n_size = noiseDim
        (lrg, lrd) = learning_rates

        # monitors
        self.gen_losses = []
        self.disc_losses = []

        # network params
        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate=lrg)
        self.optimizer_disc = tf.train.AdamOptimizer(learning_rate=lrd)
        self.gamma_reg = gamma_reg

        with tf.variable_scope(self.name):
            # placeholders
            self.n_input = tf.placeholder(tf.float32,
                                          shape=[None, self.n_size], name='gen_input')
            self.d_input = tf.placeholder(tf.float32,
                                          shape=[None, self.inDim], name='disc_input')

            # Build Generator Net
            with tf.variable_scope('generator') as gen_scope:
                self.gen_sample = generatorTemplateFxn(self.n_input,
                                                       out_dim=self.inDim,
                                                       hSeq=genSpec
                                                       )
            # Build 2 Discriminator Networks (one from noise input, one from generated samples)
            with tf.variable_scope('discriminator') as dis_scope:
                self.disc_real = discriminatorTemplateFxn(self.d_input,
                                                          hSeq=discSpec
                                                          )  # D(x)
            with tf.variable_scope(dis_scope, reuse=True):
                self.disc_fake = discriminatorTemplateFxn(self.gen_sample,
                                                          hSeq=discSpec
                                                          )  # , reuseFlag=True) # D(G(z))

            # Loss functions
            self.gen_loss = -tf.reduce_mean(tf.log(self.disc_fake))
            self.disc_loss = -tf.reduce_mean(tf.log(self.disc_real) + tf.log(1. - self.disc_fake))
            # Isolate G-vs-D variables to optimize
            self.gen_vars = [v for v in tf.trainable_variables() if "generator" in v.name]
            self.disc_vars = [v for v in tf.trainable_variables() if "discriminator" in v.name]

            # Create training operations
            self.train_gen = self.optimizer_gen.minimize(
                self.gen_loss, var_list=self.gen_vars)
            self.train_disc = self.optimizer_disc.minimize(
                self.disc_loss, var_list=self.disc_vars)
        return

    def work(self, sess, saver, num_epochs, batch_sz=96):
        '''Works through the data for given number of epochs'''
        total_steps = 0
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            # log graph for visualization
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()

            for tk in range(num_epochs):
                # Train
                gl, dl = self.train(sess, batch_sz)

                summary = tf.Summary()
                summary.value.add(tag='Losses/Generator', simple_value=float(gl))
                summary.value.add(tag='Losses/Discriminator', simple_value=float(dl))
                # summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward)) # sample perf
                self.summary_writer.add_summary(summary, tk)
                self.summary_writer.flush()

                # Periodically save episodes, model parameters, and summary statistics.
                if tk % self.autosave_every == 0:
                    print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (tk, gl, dl))
                    saver.save(sess,
                               self.model_path+'/model-'+self.name+"-v"+str(tk)+'.cptk')
                    print("Saved Model")
        return

    def train(self, sess, n):
        '''Samples a batch (size, n) of +ve exemplars.
        Then updates the GAN networks using gradients from loss;
        Generate network statistics to periodically save.'''
        inds = np.random.choice(
            len(self.dataP),
            min(n, len(self.dataP)),
            replace=False
        )
        feed_dict = {
            self.d_input: self.dataP[inds, :],
            # Generate noise to feed to the generator
            self.n_input: np.random.normal(size=[n, self.n_size])
        }
        # Train
        _, _, gl, dl = sess.run(
            [self.train_gen, self.train_disc,
             self.gen_loss, self.disc_loss],
            feed_dict=feed_dict
        )
        self.gen_losses.append(gl)
        self.disc_losses.append(dl)
        return gl, dl  # , g_norms, v_norms

    def generate(self, sess, ns):
        '''Run generator NN for samples.
        needs: num_samples to generate'''
        return sess.run(
            self.gen_sample,
            feed_dict={
                self.n_input: np.random.normal(size=[int(ns), self.n_size])
            })

    def rateSamples(self, sess, gs):
        '''Evaluate the quality of generated responses to given conditions.
        needs:
            - generated samples tensor (gs)
        gives:
            - tensor of discriminator scores (in [0,1]) for given scenarios.
        '''
        feed_dict = {self.d_input: gs}
        # Rate
        disc_ratings = sess.run(self.disc_real, feed_dict=feed_dict)
        return disc_ratings

# # Ancillary Fxns: Networks now in gan_misc.py
# def GeneratorNetwork(...):
# def DiscriminatorNetwork(...):
