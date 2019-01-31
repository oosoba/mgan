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
tensorboard --logdir=gan_worker_1:'./log/train_gan_worker_1'
'''

# from __future__ import division, absolute_import  # , print_function
import argparse, sys
import numpy as np
import scipy as sp

# ML libs
import tensorflow as tf
#import tensorflow.contrib.slim as slim

## GAN auxillary functions
sys.path.append('.')
from gan_misc import GeneratorNetwork_Cond as Generator
from gan_misc import DiscriminatorNetwork_Cond as Discriminator


# Global var defs
__version__ = "0.0.3"
tfboard_path = "./log"
class_name = "cgan_"
learning_rate = 0.0005
_eps_ = 1e-3
gamma_reg = 1e-4  # 0.001
_every_ = 10000
_noiseDim_ = 1



class Conditional_GAN():
    def __init__(self, data,
                 indices_CvG,
                 noiseDim = _noiseDim_,
                 generatorTemplateFxn=Generator,
                 discriminatorTemplateFxn=Discriminator,
                 genScope="generator",
                 genSpec=(32, 256, 64),
                 discScope="discriminator",
                 discSpec=(32, 256, 64),
                 name="worker_1",
                 model_path=tfboard_path,
                 learning_rates=(learning_rate, learning_rate)
                 ): # input_shape=(7, 7, 2),
        self.name = class_name + str(name)
        self.model_path = model_path+"/train_"+str(self.name)
        self.summary_writer = tf.summary.FileWriter(
            logdir = self.model_path
        )  # ./log/train_cgan_worker_1
        self.autosave_every = _every_
        (lrg, lrd) = learning_rates

        # Training exemplar data + dim info
        #self.class_dim, self.cand_dim, self.n_size = input_shape
        self.dataP = data
        self.n_size = noiseDim
        self.classInds, self.genInds = indices_CvG
        self.class_dim = len(indices_CvG[0])
        self.cand_dim = len(indices_CvG[1])
        self.npts, self.inDim = self.dataP.shape  ## inDim may not be gen/cand dim

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
            with tf.variable_scope(dis_scope, reuse=True):
                self.disc_fake = discriminatorTemplateFxn(self.class_input, self.gen_sample,
                                                          hSeq=discSpec)  # , reuseFlag=True) # D(G(z))

            # Loss functions
            self.gen_loss = -tf.reduce_mean(tf.log(self.disc_fake))
            self.disc_loss = -tf.reduce_mean(
                tf.log(tf.clip_by_value(self.disc_real, _eps_, 1-_eps_)) +
                tf.log(1. - tf.clip_by_value(self.disc_fake, _eps_, 1-_eps_) )
                )
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
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            # log graph for visualization
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()

            for tk in range(num_epochs):
                # sample cases & Train
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
            self.n_input: np.random.uniform(-1., 1.,size=[n, self.n_size])
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

    def generate(self, sess, c_inds):
        '''Run generator NN for samples.
        needs: class-conditional indices to generate responses to.'''
        #inds = c_inds.flatten()[...,None]
        feed_dict = {
            self.class_input: c_inds,
            self.n_input: np.random.uniform(-1., 1.,size=[len(c_inds), self.n_size])
        }
        # Train
        gsamps = sess.run(self.gen_sample, feed_dict=feed_dict)
        return gsamps
