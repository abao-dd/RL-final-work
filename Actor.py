import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from config import *
import numpy as np


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim # 动作维度
        self.action_bound = action_bound # 动作范围值,gym可以知道
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a  Actor的Online网络，产生action动作
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic Actor的Target网络，用来计算Critic更新的label yi的参数
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        #网络参数
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            #hard_replace 更新,硬更新是每隔一定回合数网络参数复制给目标网络，不可避免的，每次更新时的loss都会波动。
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            #soft_replace 更新,更新的时候不直接复制网络参数，而是每次更新参数的时候利用一个衰减的比例去逼近
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        # 搭建actor网络,输入s, 输出a = u(s), u是用网络仿真策略u函数  Build an actor network, input s, output a = u(s), u is the network simulation strategy u function
        # 单隐层，直接输出action动作 Single hidden layer, directly output action actions
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, #映射到0-1 Mapped to 0-1
                                          kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                # Scale output to -action_bound to action_bound 将(0,1)的action映射到action_bound的范围
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')

        return scaled_a   #返回动作

    def learn(self, s):   # batch update  批量更新action网络
        self.sess.run(self.train_op, feed_dict={S: s}) #将S赋值为s并传入
        #软更新
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            #硬更新
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s): #通过状态s选择a
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        #策略梯度
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        # Actor要朝着有可能获取最大Q的方向修改动作参数
        with tf.variable_scope('A_train'):
            #loss = -lr,注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))