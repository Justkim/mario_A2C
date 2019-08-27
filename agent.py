import tensorflow as tf
import numpy as np
import gym
import math
import os
import flag
import model
import architecture as policies
import mario_env as env

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def main():
    config = tf.ConfigProto()
    # Avoid warning message errors
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # # Allowing GPU memory growth
    # In some cases it is desirable
    # for the process to only allocate a subset of the available memory,
    # or to only grow the memory usage as it is needed by the process.TensorFlow provides two configuration options on the session to control this.The first
    # is the allow_growth option,
    # which attempts to allocate only as much GPU memory based on runtime allocations, it starts out allocating very little memory,
    # and as sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process.
    # config.gpu_options.allow_growth = True
    #[env.make_train_0,env.make_train_1,env.make_train_2,env.make_train_3,env.make_train_4,env.make_train_5,env.make_train_6,env.make_train_7]
    #env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0
    flag.on_desktop=True

    if flag.ON_DESKTOP:
        made_env = SubprocVecEnv([env.make_train_0,env.make_train_0])
        nsteps=1


    else:

 #       made_env=SubprocVecEnv([env.make_train_0, env.make_train_0, env.make_train_0, env.make_train_0, env.make_train_0, env.make_train_0, #env.make_train_0, env.make_train_0, env.make_train_0, env.make_train_0, env.make_train_0, #env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0,env.make_train_0])
        made_env = SubprocVecEnv([env.make_train_0,env.make_train_0])
        nsteps=128





    with tf.Session(config=config):
        model.learn(policy=policies.A2CPolicy,
                            env=made_env,
                            nsteps=nsteps,
                            total_timesteps=1000000000,
                            gamma=0.99,
                            lam = 0.95,
                            vf_coef=0.5,
                            ent_coef=0.001,
                            lr = 4e-4,
                            max_grad_norm = 0.5,
                            log_interval = 10,
                            save_interval=20,decay_rate=0.001
                            )

if __name__ == '__main__': #this is important.why?
    main()

