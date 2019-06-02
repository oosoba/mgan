'''
init: Jan2019
Author: OO
Note that each template Discriminator network should be outputing `logits`
i.e. use no activation layer (`None`) at the end.
Goals:
    - GAN generic functions (for DRY issues)
Validation:
    - re-run orig GAN classes using these as imported NN modules
'''

import sys
import numpy as np
import scipy as sp

# ML libs
import tensorflow as tf
import tensorflow.contrib.slim as slim


# Global var defs
gamma_reg = 1e-2
__version__ = "0.0.1"
__kp__ = 0.8
__rate__ = 1. - __kp__

#bias_init = tf.constant_initializer(0.4999)
wgts_init = tf.truncated_normal_initializer(stddev=0.005)
bias_init = tf.truncated_normal_initializer(mean=0.1, stddev=0.005)

# Ancillary Fxns: Networks
def GeneratorNetwork_Cond(class_in, noise,
                     out_dim,
                     hSeq=(32, 256, 64),
                     gamma_reg=gamma_reg
                     ):
    '''Custom Class-Conditional Encapsulated Generator NN model'''
    # with tf.variable_scope(scope):
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(tf.concat([class_in, noise], axis=1)),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    hidden = slim.dropout(hidden, keep_prob=__kp__, is_training=True)
    hidden = slim.stack(hidden,
                        slim.fully_connected,
                        list(hSeq[1:]),
                        activation_fn=tf.nn.sigmoid,
                        weights_regularizer=regularizer,
                        weights_initializer=wgts_init,
                        biases_initializer=bias_init
                        )
    gen = slim.fully_connected(
        hidden,
        num_outputs=out_dim,
        activation_fn=tf.nn.sigmoid,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
        )
    return gen


def DiscriminatorNetwork_Cond(class_in, gen_d_in,
                         hSeq=(32, 256, 64),
                         gamma_reg=gamma_reg
                         ):
    '''NB: Custom Class-Conditional Encapsulated Discriminator NN model'''
    # with tf.variable_scope(scope, reuse=reuseFlag):
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(tf.concat([class_in, gen_d_in], axis=1)),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    hidden = slim.dropout(hidden, keep_prob=__kp__, is_training=True)
    hidden = slim.stack(hidden,
                        slim.fully_connected,
                        list(hSeq[1:]),
                        activation_fn=tf.nn.sigmoid,
                        weights_regularizer=regularizer,
                        weights_initializer=wgts_init,
                        biases_initializer=bias_init
                        )
    disc = slim.fully_connected(
        hidden, 1,
        activation_fn=tf.nn.sigmoid,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    return disc

def GeneratorNetwork(noise,
                     out_dim,
                     hSeq=(32, 256, 64),
                     gamma_reg=gamma_reg
                     ):
    '''NB: Generic Encapsulated Generator NN model: unconditional'''
    # with tf.variable_scope(scope):
    regularizer = slim.l2_regularizer(gamma_reg)

    hidden = slim.fully_connected(
        slim.flatten(noise),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.tanh,
                            weights_regularizer=regularizer,
                            weights_initializer=wgts_init,
                            biases_initializer=bias_init
                            )

    gen = slim.fully_connected(
        hidden,
        num_outputs=out_dim,
        activation_fn=tf.nn.sigmoid,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
        )
    return gen


def DiscriminatorNetwork(test_in,
                         hSeq=(32, 256, 64),
                         gamma_reg=gamma_reg
                         ):
    '''NB: Generic Encapsulated Discriminator NN model: unconditional'''
    # with tf.variable_scope(scope, reuse=reuseFlag):
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(test_in),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.sigmoid,
                            weights_regularizer=regularizer,
                            weights_initializer=wgts_init,
                            biases_initializer=bias_init
                            )

    disc = slim.fully_connected(
        hidden, 1,
        activation_fn=tf.nn.sigmoid,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    return disc
