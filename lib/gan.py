'''
init:11Apr18
Author: OO 
Goals: 
    - Encapsulate GAN learning model
    
Monitoring:
tensorboard --logdir=gan_worker_1:'./log/train_gan_worker_1'
''' 


from __future__ import division, absolute_import #, print_function
import argparse, sys, os, re, json, csv
import numpy as np
import scipy as sp

import AFSIM_ENV as AF


# ML libs
import tensorflow as tf
import tensorflow.contrib.slim as slim


#import tensorflow.contrib.gan as tfgan
#layers = tf.contrib.layers
#ds = tf.contrib.distributions


## Global var defs
tfboard_path = "./log"
learning_rate = 0.0005



class GAN():
    def __init__(self, data, scope,
                 input_shape=(7,5), 
                 trainer = tf.train.AdamOptimizer(learning_rate=0.0005),
                 g_hDims = (512, 128),
                 d_hDims = (512, 64), 
                 name = "worker_1",
                 model_path="./log", 
                 trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
                ):
        self.name = "gan_" + str(name)
        self.model_path = model_path
        self.summary_writer = tf.summary.FileWriter(
            model_path+"/train_"+str(self.name)
        ) #./log/train_gan_worker_1
        self.autosave_every = 5000
        
        # Training data set of +ve exemplars
        self.dataP = data
        
        # monitors
        self.gen_losses = []
        self.disc_losses = []
        
        # network params
        self.c_dim, self.n_size = input_shape
        
        self.gen_hidden_dims = g_hDims
        self.disc_hidden_dims = d_hDims
        self.optimizer_gen = trainer 
        self.optimizer_disc = trainer 
        
        self.gamma_reg = 0.001
        
        with tf.variable_scope(scope):
            # placeholders
            self.n_input = tf.placeholder(tf.float32, shape=[None, self.n_size], name='input_noise')
            self.d_input = tf.placeholder(tf.float32, shape=[None, self.c_dim], name='disc_input')
            
            
            # Build Generator Net 
            self.gen_sample = self.GeneratorNetwork(self.n_input) 
            # Build 2 Discriminator Networks (one from noise input, one from generated samples)
            self.disc_real = self.DiscriminatorNetwork(self.d_input) # D(x)
            self.disc_fake = self.DiscriminatorNetwork(self.gen_sample, 
                                                       reuseFlag=True) # D(G(z))
            
            #Loss functions
            self.gen_loss = -tf.reduce_mean(tf.log(self.disc_fake)) 
            self.disc_loss = -tf.reduce_mean(tf.log(self.disc_real) + tf.log(1. - self.disc_fake))
            
            # Isolate G-vs-D variables to optimize
            # scope filtering (tf.get_collection) not working. filter by names instead...
            self.gen_vars = [v for v in tf.trainable_variables() if "generator" in v.name]
            self.disc_vars = [v for v in tf.trainable_variables() if "discriminator" in v.name]
            #print(self.gen_vars, self.disc_vars)
            
            # Create training operations
            self.train_gen = trainer.minimize(self.gen_loss, var_list=self.gen_vars)
            self.train_disc = trainer.minimize(self.disc_loss, var_list=self.disc_vars)
        return
    
    ## Ancillary Fxns: Networks
    def GeneratorNetwork(self, noise):
        with tf.variable_scope("generator"):
            regularizer = slim.l2_regularizer(self.gamma_reg)
            hidden = slim.fully_connected(
                slim.flatten(noise),
                self.gen_hidden_dims[0],
                activation_fn=tf.nn.relu,
                weights_regularizer=regularizer
            )
            hidden = slim.stack(hidden,
                slim.fully_connected,
                list(self.gen_hidden_dims[1:]),
                activation_fn=tf.nn.tanh,
                weights_regularizer=regularizer
            )
            gen = slim.fully_connected(
                hidden, self.c_dim,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(0, 0.1),
                biases_initializer=None)
        return gen
    
    def DiscriminatorNetwork(self, c_in, reuseFlag = None):
        with tf.variable_scope("discriminator", reuse=reuseFlag):
            regularizer = slim.l2_regularizer(self.gamma_reg)
            hidden = slim.fully_connected(
                slim.flatten(c_in),
                self.disc_hidden_dims[0],
                activation_fn=tf.nn.relu,
                weights_regularizer=regularizer
            )
            hidden = slim.stack(hidden,
                slim.fully_connected,
                self.disc_hidden_dims[1:],
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=regularizer
            )
            disc = slim.fully_connected(
                hidden, 1,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(0, 0.1),
                biases_initializer=None)
        return disc
    
    ## Ancillary Fxns: trainer  
    
    def sample_batch(self, n):
        '''Samples a batch (size, n) of +ve exemplars.'''
        cnt = min(n, len(self.dataP))
        inds = np.random.choice(len(self.dataP), cnt, replace=False)
        tmp = self.dataP[inds,:] 
        return tmp
    
    def work(self,datasetP, num_epochs,sess,saver,batch_sz = 96):
        '''Works through the data for given number of epochs'''
        total_steps = 0
        print ("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            
            # log graph for visualization
            self.g_writer = tf.summary.FileWriter(
                self.model_path+"/train_"+str(self.name), 
                sess.graph
            )
            self.g_writer.flush()
            
            for tk in range(num_epochs): 
                # sample +ve cases
                batch = self.sample_batch(batch_sz)
                # Train
                gl, dl = self.train(batch,sess)
                
                summary = tf.Summary()
                summary.value.add(tag='Losses/Generator', simple_value=float(gl))
                summary.value.add(tag='Losses/Discriminator', simple_value=float(dl))
                #summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward)) # sample perf
                self.summary_writer.add_summary(summary, tk)
                self.summary_writer.flush()
                                
                # Periodically save episodes, model parameters, and summary statistics.
                if tk % self.autosave_every == 0 :
                    print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (tk, gl, dl))
                    saver.save(sess,
                                    self.model_path+'/model-'+self.name+"-v"+str(tk)+'.cptk')
                    print("Saved Model")                
        return
    
    def train(self,batchP,sess):
        '''Updates the GAN networks using gradients from loss;
        Generate network statistics to periodically save;
        '''
        batch_size, _ = batchP.shape        #print(batch_size)
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, self.n_size])
        # Define input dict
        feed_dict = {
            self.d_input: batchP,
            self.n_input: z
        }
        # Train
        _, _, gl, dl = sess.run(
            [self.train_gen, self.train_disc, 
             self.gen_loss, self.disc_loss],
            feed_dict=feed_dict
        )
        self.gen_losses.append(gl)
        self.disc_losses.append(dl)
        return gl, dl #, g_norms, v_norms
    
    