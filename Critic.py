import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from config import *

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim  #状态维度
        self.a_dim = action_dim #动作维度
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Critic的Online网络，用来产生Q(s,a)
            # self.a由Online Net输出
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # 停止critic网络更新流向actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            # Critic的Target网络，用来产生计算Critic更新的label yi的参数
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net
            #网络参数
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        #初始化参数
        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        # MSE 最小化这个Loss
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            #硬更新
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            #软更新
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        # 搭建Critic网络，输入s,a, 输出当前状态s下，使用a时的Q值 Build a Critic network, input s, a, and output the Q value when using a in the current state s
        # trainable = True if Reuse is None else False
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)  #正态分布 Normal distribution
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_}) #训练，更新Q网络，更新网络参数
        #通常使用软更新
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1