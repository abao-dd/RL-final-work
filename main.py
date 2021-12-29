"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import matplotlib.pyplot as plt
from Actor import *
from Critic import  *
from memory import  *

#如果需要每次试验得到相同结果，则释放注释
# np.random.seed(0)
# tf.set_random_seed(0)

#全局变量
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor Actor学习率
LR_C = 0.001    # learning rate for critic Critic学习率
GAMMA = 0.9     # reward discount 奖励衰减系数
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000 # 经验池，存储10000条(si,ai,ri,si+1)后开始训练
BATCH_SIZE = 32 # 每次从经验池中随机抽取N=32条Transition

RENDER = False # gym可视化使能False
OUTPUT_GRAPH = True

if __name__ == '__main__':

    sess = tf.Session()

    # Create actor and critic.
    # They are actually connected to each other, details can be seen in tensorboard or in this picture:
    actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
    critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    sess.run(tf.global_variables_initializer())

    M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

    # 写入日志
    # if OUTPUT_GRAPH:
    #     tf.summary.FileWriter("logs/", sess.graph)

    var = 3  # 控制探索度 前几次更新可以多进行探索，后面就少一点，因为越来越接近你的目标
    #记录reward，用于建图
    reward_buffer = []
    t1 = time.time()  # 开始时间
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):  # 时间步

            if RENDER:
                env.render()

            # # 添加探索噪声，说不定会有更好的Q值
            a = actor.choose_action(s)
            # 给action selection添加探索随机性
            a = np.clip(np.random.normal(a, var), -2, 2)    # 范围-2,2,在正态分布中概率取值
            s_, r, done, info = env.step(a)

            M.store_transition(s, a, r / 10, s_)

            if M.pointer > MEMORY_CAPACITY:
                var *= .9995    # 衰减你的探索者，到后面，就不需要在取探索了
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]  #从bt获得数据s
                b_a = b_M[:, state_dim: state_dim + action_dim] # 从bt获得数据a
                b_r = b_M[:, -state_dim - 1: -state_dim] # 从bt获得数据r
                b_s_ = b_M[:, -state_dim:] # 从bt获得数据s'

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                reward_buffer.append(ep_reward)
                if ep_reward > -100: #奖励大于-100，可以可视化
                    RENDER = True
                break

    print('Running time: ', time.time()-t1)

    plt_episode = np.arange(0, MAX_EPISODES, 1)
    plt.plot(plt_episode, reward_buffer, '>--', color='red', label='reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()