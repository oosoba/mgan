from gan_misc import DiscriminatorNetwork
from gan_misc import GeneratorNetwork
'''
init:11May18
revamp: Jan2019
Author: OO
Goals:
    - Generalize GAN model to discriminator auction model
    - Update BP procedures to use NEM noise

Monitoring:
tensorboard --logdir=gAuction_1:'./log/train_gan_auction_worker_1'
'''

# from __future__ import division, absolute_import  # , print_function
import argparse
import sys
import numpy as np
import scipy as sp
from collections import defaultdict

# ML libs
import tensorflow as tf
#import tensorflow.contrib.slim as slim

# GAN auxillary functions
sys.path.append('.')

# Global var defs
__version__ = "0.0.3"
tfboard_path = "./log"
class_name = "gan_auction_"

gamma_reg = 1e-4  # 0.001
_every_ = 5000
_noiseDim_ = 4

#aggregates = defaultdict()
aggregators = {
    "avg": tf.log,
    "max": (lambda sa: tf.reduce_max(tf.log(sa), axis=1)),
    "min": (lambda sa: tf.reduce_min(tf.log(sa), axis=1))
}
__agg_opts__ = aggregators.keys()
# careful with dict in py3, need to list keys/values


class GANAuction():
    def __init__(self, data,
                 auction_size=3,
                 noiseDim=_noiseDim_,
                 generatorTemplateFxn=GeneratorNetwork,
                 discriminatorTemplateFxn=DiscriminatorNetwork,
                 genScope="generator",
                 genSpec=(32, 256, 64),
                 discScope="discriminator",
                 discSpec=(32, 256, 64),
                 aggregator="avg",
                 name="worker_1",
                 model_path=tfboard_path,
                 learning_rates=(2e-4, 2e-4)
                 ):
        self.name = class_name + str(name)
        self.model_path = model_path+"/train_"+str(self.name)
        self.summary_writer = tf.summary.FileWriter(
            logdir=self.model_path
        )
        self.autosave_every = _every_
        self.aggregatorFxn = aggregators[aggregator] if aggregator in aggregators.keys(
        ) else aggregators["avg"]
        # Training exemplar data + dim info
        self.dataP = data
        self.npts, self.inDim = self.dataP.shape
        self.n_size = noiseDim  # 64
        (lrg, lrd) = learning_rates

        # monitors
        self.g_losses = []
        self.d_losses = []

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

            # instantiate panel of experts
            dscope_names = ["discriminator_" + str(t) for t in range(auction_size)]

            # Build 2 Discriminator Nets per expert:
            # one from noise input, one from generated samples
            # experts still identically spec'd (via discSpec)...
            self.disc_reals = {}
            self.disc_fakes = {}
            for dsn in dscope_names:
                with tf.variable_scope(dsn) as dis_scope:
                    self.disc_reals[dsn] = discriminatorTemplateFxn(self.d_input,
                                                                    hSeq=discSpec
                                                                    )  # D_i(x)
                with tf.variable_scope(dis_scope, reuse=True):
                    self.disc_fakes[dsn] = discriminatorTemplateFxn(self.gen_sample,
                                                                    hSeq=discSpec
                                                                    )  # D_i(G(z))

            # Auction process
            with tf.name_scope("auctioning"):
                # gather discriminator bids...
                self.auction = tf.stack(
                    list(self.disc_fakes.values()),
                    axis=1
                )
                # selection...
                self.decisions = tf.map_fn(
                    self.aggregatorFxn,
                    self.auction
                )

            # Gen: Loss, Vars, Trainer
            with tf.name_scope("gLoss"):
                self.gen_loss = -tf.reduce_mean(self.decisions)

            with tf.name_scope("gTrain"):
                self.gen_vars = [v for v in tf.trainable_variables() if "generator" in v.name]
                self.train_gen = self.optimizer_gen.minimize(self.gen_loss, var_list=self.gen_vars)

            # Disc: Losses, Vars, Trainers
            self.disc_losses = {}
            self.disc_vars = {}
            self.train_discs = {}
            ct = 0
            for dn in dscope_names:
                self.disc_vars[dn] = [v for v in tf.trainable_variables() if dn in v.name]
                with tf.name_scope("dLoss_"+str(ct)):
                    self.disc_losses[dn] = -tf.reduce_mean(
                        tf.log(self.disc_reals[dn]) + tf.log(1. - self.disc_fakes[dn])
                    )  # -tf.reduce_mean(tf.log(self.disc_real) + tf.log(1. - self.disc_fake))
                with tf.name_scope("dTrain_"+str(ct)):
                    self.train_discs[dn] = self.optimizer_disc.minimize(
                        self.disc_losses[dn], var_list=self.disc_vars[dn]
                    )
                ct += 1
        return

    def work(self, sess, saver, num_epochs, batch_sz=512):
        '''Works through the data for given number of epochs'''
        total_steps = 0
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            # log graph for visualization
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()

            for tk in range(num_epochs):
                # Train on dataP
                self.train(sess, batch_sz)

                summary = tf.Summary()
                summary.value.add(tag='Generator/Loss',
                                  simple_value=float(self.g_losses[-1][0]))
                summary.value.add(tag='Discriminator Panel/Max Loss',
                                  simple_value=float(np.max(self.d_losses[-1])))
                summary.value.add(tag='Discriminator Panel/Avg. Loss',
                                  simple_value=float(np.mean(self.d_losses[-1])))
                summary.value.add(tag='Discriminator Panel/Min Loss',
                                  simple_value=float(np.min(self.d_losses[-1])))
                self.summary_writer.add_summary(summary, tk)
                self.summary_writer.flush()
                # Periodically save episodes, model parameters, and summary statistics.
                if tk % self.autosave_every == 0:
                    self.model_save(tk, sess, saver)
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
        sess.run(
            [self.train_gen],
            feed_dict=feed_dict
        )
        sess.run(
            list(self.train_discs.values()),
            feed_dict=feed_dict
        )
        gl = sess.run(
            [self.gen_loss],
            feed_dict=feed_dict
        )
        dls = sess.run(
            list(self.disc_losses.values()),
            feed_dict=feed_dict
        )
        self.g_losses.append(gl)
        self.d_losses.append(dls)
        return gl, dls

    def model_save(self, tk, sess, saver):
        print('Step, Generator Loss, Discriminator Losses:',
              (tk,
               np.mean(self.g_losses[-self.autosave_every:]),
               np.mean(self.d_losses[-self.autosave_every:], axis=0)
               )
              )
        if not any(np.isnan(self.g_losses[-self.autosave_every:])):
            saver.save(sess, self.model_path+'/model-'+self.name+"-v"+str(tk)+'.cptk')
            print("Saved Model")
        else:
            print("Model issues... Not saved!")
        return

    def generate(self, sess, ns):
        '''Run generator NN for samples. (still assuming single gen)
        needs: class indices to generate'''
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
        [one column per discriminator]
        '''
        feed_dict = {self.d_input: gs}
        # Rate
        disc_ratings = sess.run(
            list(self.disc_reals.values()),
            feed_dict=feed_dict
        )
        return disc_ratings

# Snip.
# Auction process....
# self.gen_loss = -tf.reduce_mean(tf.log(self.auction))
# self.gen_loss = -tf.reduce_mean(
#     tf.reduce_max(tf.log(self.auction), axis=1)
#     ) # pick max D(x_i); fool at least one
# self.gen_loss = -tf.reduce_mean(
#    tf.reduce_min(tf.log(self.auction), axis=1)
#    ) # pick min D(x_i); fool all D's

# # Ancillary Fxns: Networks now in gan_misc.py
# def GeneratorNetwork(...):
# def DiscriminatorNetwork(...):
