import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import copy
from XFP import XFP, LeducRLEnv
from dataset import Dataset


class PPO(object):
    """ if current policy is p=[p1, p2], then B(p)=[br(p2), br(p1)]"""
    def __init__(self, flags, player):
        self.flags = flags
        self.player = player  # 0 or 1
        self.opponent_policy = None
        self.policy = None  # explicit policy
        self.value = None  # explicit value
        np.random.seed(flags.seed)
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=flags.num_cpu,
            intra_op_parallelism_threads=flags.num_cpu)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        with self.graph.as_default():
            tf.set_random_seed(flags.seed)
            self.env = LeducRLEnv(card_num=flags.card_num, seed=flags.seed)

            # build inference
            # policy and v
            self.state_history = tf.placeholder(tf.int32, [None] + self.env.state_history_space, 'state_history')
            self.state_card = tf.placeholder(tf.int32, [None] + self.env.state_card_space, 'state_card')
            state_history = tf.reshape(tf.cast(self.state_history, tf.float32), [-1] + [reduce(lambda x, y: x * y, self.env.state_history_space)])
            state_card = tf.cast(self.state_card, tf.float32)
            self.state = tf.concat([state_card, state_history], axis=1)

            with tf.variable_scope('current'):
                self.pi_logits, self.v = self._policy_v()
            self.pi_softmax = tf.nn.softmax(self.pi_logits)  # for test
            with tf.variable_scope('old'):
                self.pi_logits_old, self.v_old = self._policy_v()

            # build training
            self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.action = tf.placeholder(tf.int32, [None])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.ret = tf.placeholder(tf.float32, [None])  # empirical return
            one_hot_actions = tf.one_hot(self.action, self.env.action_space, dtype=tf.float32)

            self.neglog_pi = tf.nn.softmax_cross_entropy_with_logits(logits=self.pi_logits, labels=one_hot_actions)
            self.log_pi = -self.neglog_pi
            self.neglog_pi_old = tf.nn.softmax_cross_entropy_with_logits(logits=self.pi_logits_old, labels=one_hot_actions)
            self.log_pi_old = -self.neglog_pi_old

            # entropy
            a0 = self.pi_logits - tf.reduce_max(self.pi_logits, axis=-1, keep_dims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
            p0 = ea0 / z0
            self.pi_entropy = tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
            self.mean_pi_entropy = tf.reduce_mean(self.pi_entropy)

            # kl(pi_new, pi_old)
            a1= self.pi_logits_old - tf.reduce_max(self.pi_logits_old, axis=-1, keep_dims=True)
            ea1 = tf.exp(a1)
            z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
            self.kl = tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
            self.mean_kl = tf.reduce_mean(self.kl)

            ratio = tf.exp(self.log_pi - tf.stop_gradient(self.log_pi_old))  # pi_new / pi_old
            surr1 = ratio * self.advantage
            surr2 = tf.clip_by_value(ratio, 1.0 - flags.ppo_clip, 1.0 + flags.ppo_clip) * self.advantage
            self.surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))  # maximize surrogate objective
            # self.surr_loss = -tf.reduce_mean(surr1)  # maximize surrogate objective

            self.vf_loss = tf.reduce_mean(tf.square(self.v - self.ret))  # minimize value loss

            self.entropy_loss = - flags.encoeff * tf.reduce_mean(self.pi_entropy)  # maximize entropy
            total_loss = self.surr_loss + self.vf_loss + self.entropy_loss

            # self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            grad_var_list = self.optimizer.compute_gradients(total_loss)
            self.train_grad_var_list = []
            for i in range(len(grad_var_list)):
                if grad_var_list[i][0] is None: continue
                if grad_var_list[i][1].name.find('linear1') != -1:
                    self.train_grad_var_list.append((grad_var_list[i][0]/2, grad_var_list[i][1]))
                else:
                    self.train_grad_var_list.append(grad_var_list[i])

            self.apply_gradients = self.optimizer.apply_gradients(self.train_grad_var_list, self.global_step)

            with tf.name_scope('copy4old'):
                assign_ops = []
                for (cur, old) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'),
                                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old')):
                    assert cur.name[cur.name.rfind('/') + 1:] == old.name[old.name.rfind('/') + 1:]
                    assign_ops.append(tf.assign(old, cur))
                self.copy_cur2old_op = tf.group(*assign_ops)

            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)
            self.sess.run(self.copy_cur2old_op)

        self.data_states_card = np.zeros([self.flags.data_len] + self.env.state_card_space, np.int32)
        self.data_states_history = np.zeros([self.flags.data_len] + self.env.state_history_space, np.int32)
        self.data_actions = np.zeros([self.flags.data_len], np.int32)
        self.data_returns = np.zeros([self.flags.data_len], np.float32)
        self.data_advant = np.zeros([self.flags.data_len], np.float32)
        self.top = 0
        self.episode_start_index = 0

    def _linear_layer(self, linear_in, dim, hiddens):
        weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                  initializer=tfc.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True))
        bias = tf.get_variable('bias', [hiddens], tf.float32,
                               initializer=tf.constant_initializer(0.1))
        pre_activations = tf.add(tf.matmul(linear_in, weights), bias)
        linear_out = tf.nn.relu(pre_activations)
        return linear_out

    def _policy_v(self):
        with tf.variable_scope('linear1'):
            dim = self.state.get_shape().as_list()[1]; hiddens = 48
            linear1 = self._linear_layer(self.state, dim, hiddens)
        with tf.variable_scope('policy'):
            with tf.variable_scope('linear2'):
                dim = hiddens; hiddens = 32
                linear2 = self._linear_layer(linear1, dim, hiddens)
            with tf.variable_scope('linear3'):
                dim = hiddens; hiddens = self.env.action_space
                pi_logits = self._linear_layer(linear2, dim, hiddens)

        with tf.variable_scope('linear1'):
            tf.get_variable_scope().reuse_variables()
            dim = self.state.get_shape().as_list()[1]; hiddens = 48
            linear1 = self._linear_layer(self.state, dim, hiddens)
        with tf.variable_scope('value'):
            with tf.variable_scope('linear2'):
                dim = hiddens; hiddens = 32
                linear2 = self._linear_layer(linear1, dim, hiddens)
            with tf.variable_scope('linear3'):
                dim = hiddens; hiddens = 1
                v = self._linear_layer(linear2, dim, hiddens)
        return pi_logits, v

    def compute_self_policy(self, verbose=False):
        policy = {}
        value = {}
        for cards in XFP.possible_cards_list:
            for history in LeducRLEnv.history_string2vector:
                state_card = np.zeros([self.env.card_num], np.int32)
                state_history = LeducRLEnv.history_string2vector[history]
                pround = 1 if history in XFP.round1_states_set else 2
                if history in XFP.player1_states_set:
                    if self.player == 1 and not verbose: continue
                    state_card[int(cards[0])] = 1
                    card = cards[0]
                    if pround == 2:
                        state_card[int(self.env.card_num) / 2 + int(cards[1])] = 1
                        card = cards[:2]
                else:
                    if self.player == 0 and not verbose: continue
                    state_card[int(cards[2])] = 1
                    card = cards[2]
                    if pround == 2:
                        state_card[int(self.env.card_num) / 2 + int(cards[1])] = 1
                        card = cards[1:]

                nn_pi = self.sess.run([self.pi_softmax, self.v],
                                                     feed_dict={self.state_card: [state_card],
                                                                self.state_history: [state_history]})
                policy[card + history] = nn_pi[0][0]
                value[card + history] = nn_pi[1][0]
        self.policy = policy
        self.value = value
        return policy, value

    def learn(self, policy_gradient_train_num, num_games, opponent_policy):
        # record the best
        best_behaved_policy = None
        best_behaved_policy_payoff = -10000

        self.opponent_policy = opponent_policy
        self.compute_self_policy()
        learning_rate = self.flags.lr
        last_pg_payoff = -10000.0
        for _ in range(policy_gradient_train_num):
            self.collect_samples(num_games)
            print "player {:d} iter {:d} : training from {:d} games, totally {:d} transitions, lr={:.7f}".format(
                self.player, _, num_games, self.top, learning_rate)
            self.sess.run(self.copy_cur2old_op)
            d = Dataset({'card': self.data_states_card,
                         'history': self.data_states_history,
                         'action': self.data_actions,
                         'return': self.data_returns,
                         'advantage': self.data_advant}, self.top)

            for epoch in range(self.flags.epochs):
                print "  {: ^13}|{: ^13}|{: ^13}|{: ^13}|{: ^13}".format("meanKL", "meanEntropy", "surr_loss", "vfloss", "entro_loss")
                print_times = 0
                for batch in d.iterate_once(self.flags.batch):
                    train_res = self.sess.run([self.mean_kl,
                                               self.mean_pi_entropy,
                                               self.surr_loss,
                                               self.vf_loss,
                                               self.entropy_loss,
                                               self.apply_gradients], feed_dict={
                        self.state_card: batch['card'],
                        self.state_history: batch['history'],
                        self.action: batch['action'],
                        self.ret: batch['return'],
                        self.advantage: batch['advantage'],
                        self.lr: learning_rate
                    })

                    if print_times % int(self.top/self.flags.batch/2) == 0:
                        print "  {: ^13.4f}|{: ^13.4f}|{: ^13.4f}|{: ^13.4f}|{: ^13.4f}".format(*train_res[:-1])
                    print_times += 1

            self.compute_self_policy()
            if self.player == 0:
                realization = XFP.compute_realization(self.policy, self.opponent_policy)
            else:
                realization = XFP.compute_realization(self.opponent_policy, self.policy)
            payoff = XFP.compute_payoff_given_realization(realization)
            print 'player {:d} iter {:d} achieves payoff [{:.4f} {:.4f}]'.format(
                self.player,
                _,
                *payoff)
            if last_pg_payoff > payoff[self.player] or np.isclose(last_pg_payoff, payoff[self.player], atol=0.0001):
                learning_rate = learning_rate * 0.6
            last_pg_payoff = payoff[self.player]
            if payoff[self.player] > best_behaved_policy_payoff:
                best_behaved_policy_payoff = payoff[self.player]
                best_behaved_policy = copy.deepcopy(self.policy)
        return best_behaved_policy, best_behaved_policy_payoff

    def collect_samples(self, num_games):
        self.top = 0
        self.episode_start_index = 0
        for game in xrange(num_games):
            self.episode_start_index = self.top
            ob = self.env.reset()
            while ob['turn'] != -1:
                state_str = ob['card_str'] + ob['history_str']
                if ob['turn'] == self.player:
                    action = np.random.choice(2, [], p=self.policy[state_str])
                    value = self.value[state_str]
                    self.data_states_card[self.top] = ob['card']
                    self.data_states_history[self.top] = ob['state']
                    self.data_actions[self.top] = action
                    self.data_advant[self.top] = value
                    self.top += 1
                    assert self.top < self.flags.data_len
                else:
                    action = np.random.choice(2, [], p=self.opponent_policy[state_str])
                ob = self.env.act(action)
            episode_return = ob['payoff'][self.player]
            while self.episode_start_index != self.top:
                self.data_returns[self.episode_start_index] = episode_return
                self.data_advant[self.episode_start_index] = episode_return - self.data_advant[self.episode_start_index]
                self.episode_start_index += 1
