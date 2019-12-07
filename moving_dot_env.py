import gym_moving_dot

# Part taken from adborghi fantastic implementation
# https://github.com/aborghi/retro_contest_agent/blob/master/fastlearner/ppo2ttifrutti_sonic_env.py
import numpy as np
import gym
from baselines.common.atari_wrappers import FrameStack
import flag
import collections

from baselines.common.distributions import make_pdtype


# import gym_remote.client as grc


# This will be useful for stacking frames
# from baselines.common.atari_wrappers import FrameStack

# Library used to modify frames (former times we used matplotlib)
import cv2

# setUseOpenCL = False means that we will not use GPU (disable OpenCL acceleration)
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plot


class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
        - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # Set frame to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1


        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        # if flag.DEBUG:
        #     cv2.imshow("frame",frame)
        #     cv2.waitKey(0)
        return frame



class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):


        return reward


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.reward_q = collections.deque()


    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        # print("rew",rew)
        # self._cur_x += rew
        # rew = max(0, self._cur_x - self._max_x)
        # self._max_x = max(self._max_x, self._cur_x)
        # # print("real reward",rew)
        stuck_flag =True



        rew=(rew) /15  - 0.01
       # rew=(rew) /15 -1
        # rew=rew - 1
        #if(len(self.reward_q)==10):
         #   self.reward_q.popleft()
        #self.reward_q.append(rew)

       # for i in list(self.reward_q):

            #if i > -0.01:
             #   stuck_flag = False
       # if stuck_flag:
        #    rew=rew - 1.01

        return obs,rew, done, info


def make_env(env_idx):
    """
    Create an environment with some standard wrappers.
    """


    # Make the environment


    #levelList = ['SuperMarioBros-1-1-v2','SuperMarioBros-2-1-v0','SuperMarioBros-3-1-v0','SuperMarioBros-4-1-v0','SuperMarioBros-5-1-v0','SuperMarioBros-6-1-v0','SuperMarioBros-7-1-v0','SuperMarioBros-8-1-v0']


    # record_path = "./records/" + dicts[env_idx]['state']

    env = gym.make("MovingDot-v0")
    #SuperMarioBros-v0
    #SuperMarioBrosRandomStages
    # env = BinarySpaceToDiscreteSpaceEnv(env,RIGHT_ONLY)

    #env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    # env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)


    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    #env = AllowBacktracking(env)
    return env


def make_train_0():
    new_env=make_env(0)
    # new_env = gym.wrappers.Monitor(new_env, "recording0")
    return new_env


def make_train_1():
    new_env = make_env(1)
    #new_env = gym.wrappers.Monitor(new_env, "recording1")
    return new_env


def make_train_2():
    new_env = make_env(2)
    #new_env = gym.wrappers.Monitor(new_env, "recording2")
    return new_env


def make_train_3():
    new_env = make_env(3)
    #new_env = gym.wrappers.Monitor(new_env, "recording3")
    return new_env


def make_train_4():
    new_env = make_env(4)
    #new_env = gym.wrappers.Monitor(new_env, "recording4")
    return new_env


def make_train_5():
    new_env = make_env(5)
    #new_env = gym.wrappers.Monitor(new_env, "recording5")
    return new_env


def make_train_6():
    new_env = make_env(6)
    #new_env = gym.wrappers.Monitor(new_env, "recording6")
    return new_env


def make_train_7():
    new_env = make_env(7)
    #new_env = gym.wrappers.Monitor(new_env, "recording7")
    return new_env


def make_train_8():
    new_env = make_env(8)
    # new_env = gym.wrappers.Monitor(new_env, "recording8")
    return new_env


def make_train_9():
    return make_env(9)


def make_train_10():
    return make_env(10)


def make_train_11():
    return make_env(11)


def make_train_12():
    return make_env(12)


def make_test_level_Green():
    return make_test()


def make_test():
    """
    Create an environment with some standard wrappers.
    """

    # Make the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
    print(env.action_space)
    # Build the actions array
    # env = ActionsDiscretizer(env)

    # Scale the rewards
    # env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    # env = AllowBacktracking(env)

    return env





#
# new_env=make_env(0)
# new_env.reset()
# for i in range(0,80):
#
#     new_env.step(2)
#     new_env.render()
#
#     time.sleep(0.05)

#
# while True:
#    frame=[]
#    a,b,c,d=new_env.step(6)
#    print("reward is",b)
#    # new_env.step(3)
#    # new_env.step(1)
#    # new_env.step(5)
#    # a,b,c,d=new_env.step(6)
#    # print(b)
#    new_env.render()
# #
#    time.sleep(0.05)
# action_space=new_env.action_space
#
# pdtype = make_pdtype(action_space)
#
#
# while True:
#    a0= pdtype.sample()
#
#    a,b,c,d=new_env.step(a0)
#    print("action",a0)
#    # new_env.step(3)
#    # new_env.step(1)
#    # new_env.step(5)
#    # a,b,c,d=new_env.step(6)
#    # print(b)
#    new_env.render()
#
#    time.sleep(0.05)






