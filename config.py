# all placeholder for tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

ENV_NAME = 'Pendulum-v1'  #倒立摆


env = gym.make(ENV_NAME) # 选择倒立摆环境
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0] # 获得state维度
action_dim = env.action_space.shape[0] # 获得action维度
action_bound = env.action_space.high # 获得动作范围action_bound

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')