import numpy as np
import tensorflow as tf
import baselines.common.distributions as dp
import flag

import baselines.common.distributions
from baselines.common.distributions import make_pdtype


# This function selects the probability distribution over actions
from baselines.common.distributions import make_pdtype


# Convolution layer
def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())


# Fully connected layer
def fc_layer(inputs, units, activation_fn=tf.nn.elu, gain=1.0,name1="defualt"):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name1
                           )

"""
This object creates the A2C Network architecture
"""


class A2CPolicy(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse=False):
        # This will use to initialize our kernels
        gain = np.sqrt(2)


        # Based on the action space, will select what probability distribution type
        # we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType
        # aka Diagonal Gaussian, 3D normal distribution

        self.pdtype = dp.CategoricalPdType(7)


        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)

        # Create the input placeholder
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        # Normalize the images
        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        """
        Build the model
        3 CNN for spatial dependencies
        Temporal dependencies is handle by stacking frames
        (Something funny nobody use LSTM in OpenAI Retro contest)
        1 common FC
        1 FC for policy
        1 FC for value
        """
        with tf.variable_scope("model", reuse=reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)#8 8 4 32
            conv2 = conv_layer(conv1, 64, 4, 2, gain) #4 4 32 64
            conv3 = conv_layer(conv2, 64, 3, 1, gain)# 8 64 8 8
            flatten1 = tf.layers.flatten(conv3,"FLAT") # 8 4096
            self.fc_common = fc_layer(flatten1, 512,gain=gain,name1="FC512") # 8 512

            #
            # self.p_layer = tf.layers.dense(inputs=self.fc_common,
            #                                units=7,
            #                                activation=tf.nn.elu,
            #                                kernel_initializer=tf.orthogonal_initializer(gain=0.01), name="p_layer"
            #                                )
            # This build a fc connected layer that returns a probability distribution
            # over actions (self.pd) and our pi logits (self.pi).
            # self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)
            if flag.LAST_LAYER_IMPL:
                self.p_layer = tf.layers.dense(inputs=self.fc_common,
                                               units=8,
                                               activation=tf.nn.elu,
                                               kernel_initializer=tf.orthogonal_initializer(gain=0.01), name="p_layer"
                                               )
                self.softmax_layer=tf.nn.softmax(self.p_layer,name="softmax")
                self.dist = tf.distributions.Categorical(probs=self.softmax_layer)
            #
                a0=self.dist.sample()
            else:
                self.pdtype = make_pdtype(action_space)
                self.pd, self.pi = self.pdtype.pdfromlatent(self.fc_common, init_scale=0.01)
                a0 = self.pd.sample()

            # entropy=self.dist.entropy("entropy")



            #tf.clip_by_value(fc_common,1e-10,1.0)-

            #self.pd, self.pi = self.pdtype.pdfromlatent(self.fc_common, init_scale=0.01)
            # Calculate the v(s)


            vf = fc_layer(self.fc_common, 1, activation_fn=None,name1="LASTFC")[:,0] #8 1

            #a0 = self.pd.sample()
        self.initial_state = None

        # Take an action in the action distribution (remember we are in a situation
        # of stochastic policy so we don't always take the action with the highest probability
        # for instance if we have 2 actions 0.7 and 0.3 we have 30% chance to take the second)



        # Function use to take a step returns action to take and V(s)
        def step(state_in, *_args, **_kwargs):
            # sl,action, value= sess.run([self.softmax_layer,a0, vf], {inputs_: state_in})
            action, value = sess.run([a0, vf], {inputs_: state_in})
            #print(sl)

            # max=-1
            # maxa=0
            # for i in range(0,7):
            #     if max<pi[0][i]:
            #         max=pi[0][i]
            #         maxa=i
            # print("NOOP: ",pi[0][1])
            # print("RIGHT: ",pi[0][1])
            # print("RIGHTJUMP: ",pi[0][2])
            # print("RIGHTFAST: ",pi[0][3])
            # print("RIGHTJUMPFAST: ",pi[0][4])
            # print("JUMP:",pi[0][5])
            # print("LEFT: ",pi[0][6])






            return action, value

        # Function that calculates only the V(s)
        def value(state_in, *_args, **_kwargs):
            return sess.run(vf, {inputs_: state_in})

        # Function that output only the action to take
        def select_action(state_in, *_args, **_kwargs):
            return sess.run(a0, {inputs_: state_in})

        self.inputs_ = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action
