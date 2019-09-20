import numpy as np
import tensorflow as tf
import baselines.common.distributions as dp
import flag
#import secrets

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






        height, weight, channel = ob_space.shape

        ob_shape = (height, weight, channel)
        epsilon_=tf.placeholder(tf.float32)


        # Create the input placeholder
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        self.pre_actions_=tf.placeholder(tf.float32,[None,7],name="pre_actions_")

        # Normalize the images
        scaled_images = tf.cast(inputs_, tf.float32) / 255.0


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

            self.flatten1 = tf.layers.flatten(conv3,"FLAT") # 8 4096

            self.fc_common = fc_layer(self.flatten1, 512,gain=gain,name1="FC512")




            # print_out=tf.print(self.fc_common,[self.fc_common],"fc_common")



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
                                               units=7,
                                               activation=tf.nn.elu,
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name="p_layer"
                                               )

                self.p_layer=(tf.maximum((self.p_layer ), 1e-13))
                if flag.CONCAT_LAYER:
                    self.p_layer = tf.concat([self.p_layer, self.pre_actions_], 1)
                self.softmax_layer = tf.nn.softmax(self.p_layer, name="softmax")
                self.dist = tf.distributions.Categorical(logits=self.p_layer)

                #self.dist=tf.contrib.distributions.MultivariateNormalDiag(logits=self.p_layer)



                if not flag.USE_ARGMAX:
                    a0=self.dist.sample()
                else:
                    a0=tf.argmax(self.softmax_layer,axis = 1)



                self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.p_layer, labels=a0)
            else:
                self.pdtype = dp.DiagGaussianPdType(7)
                #self.pdtype = make_pdtype(action_space)
                self.pd, self.pi = self.pdtype.pdfromlatent(self.fc_common, init_scale=0.01)
                a0 = self.pd.sample()
                print(self.pi.shape)
                print(a0)
                self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pi, labels=a0)



            #random = tf.random_uniform(shape=(),dtype=tf.float32)
            #random_uniform=tf.random_uniform(shape=([1]), minval=0, maxval=6, dtype=tf.int32)



            #a0=tf.cond(random<epsilon_,lambda :random_uniform,lambda:sample_action)


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
        #stupied idea:))





        # Function use to take a step returns action to take and V(s)
        def step(state_in,epsilon,pre_actions, *_args, **_kwargs):


            # sl,action, value= sess.run([self.softmax_layer,a0, vf], {inputs_: state_in})
            actions,value,neglogpac,pi,flatLayer ,inputs1= sess.run([a0,vf, self.neglogpac,self.softmax_layer,scaled_images,inputs_], {inputs_: state_in,epsilon_:epsilon,self.pre_actions_:pre_actions})
            # print("action_shape",actions.shape)
            # print("flatten",flatLayer.shape)
            # print("sc",sc.shape)
            # print("inputs",inputs1.shape)



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





            return actions, value,neglogpac,pi

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
