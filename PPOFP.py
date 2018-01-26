import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import copy
from XFP import XFP, LeducRLEnv


class PPO(object):
    """ if current policy is p=[p1, p2], then B(p)=[br(p2), br(p1)]"""
    def __init__(self, flags):
        self.flags = flags
        self.explicit_policy = None  # explicit policy only possible to be evaluated in small games
        self.explicit_value = None  # explicit value
        self.opponent_policy = None
        np.random.seed(flags.seed)
        self.env = LeducRLEnv(card_num=flags.card_num, seed=flags.seed)

        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=flags.num_cpu,
            intra_op_parallelism_threads=flags.num_cpu)
        self.device = '/gpu:0' if self.flags.use_gpu else '/cpu:0'
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        self.sess.__enter__()
        tf.set_random_seed(flags.seed)

        with tf.device(self.device):
            self.lr = tf.get_variable('learning_rate', [], tf.float32, tf.constant_initializer(self.flags.lr), trainable=False)
            self.ops = [{}, {}]  # ops for position1 and position2

            for i in range(2):
                with tf.variable_scope("player"+str(i)):
                    # build data pipeline
                    self.ops[i]['state_history'] = tf.placeholder(tf.int32, [None] + self.env.state_history_space, 'state_history')
                    self.ops[i]['state_card'] = tf.placeholder(tf.int32, [None] + self.env.state_card_space, 'state_card')
                    self.ops[i]['action'] = tf.placeholder(tf.int32, [None])
                    self.ops[i]['advantage'] = tf.placeholder(tf.float32, [None])
                    self.ops[i]['ret'] = tf.placeholder(tf.float32, [None])  # empirical return
                    dataset = tf.data.Dataset.from_tensor_slices((self.ops[i]['state_history'],
                                                                  self.ops[i]['state_card'],
                                                                  self.ops[i]['action'],
                                                                  self.ops[i]['advantage'],
                                                                  self.ops[i]['ret']))
                    dataset = dataset.shuffle(10000)
                    dataset = dataset.batch(self.flags.batch)
                    dataset = dataset.repeat(self.flags.epochs)
                    iterator = dataset.make_initializable_iterator()
                    next_element = iterator.get_next()
                    self.ops[i]['iterator'] = iterator
                    self.ops[i]['next_element'] = next_element

                    # build training
                    self.ops[i]['pi_logits'], self.ops[i]['v'], self.ops[i]['pi_logits_old'], self.ops[i]['v_old'], _ = \
                        self._build_inference(next_element[0], next_element[1])
                    self.ops[i]['mean_kl'], self.ops[i]['mean_pi_entropy'], self.ops[i]['surr_loss'], \
                        self.ops[i]['vf_loss'], self.ops[i]['apply_gradients'], self.ops[i]['copy'] = \
                        self._build_train(next_element[2], next_element[3], next_element[4], self.ops[i]['pi_logits'],
                                          self.ops[i]['pi_logits_old'], self.ops[i]['v'])

                    # build inference
                    self.ops[i]['state_history_inf'] = tf.placeholder(tf.int32, [None] + self.env.state_history_space,
                                                                      'state_history_inf')
                    self.ops[i]['state_card_inf'] = tf.placeholder(tf.int32, [None] + self.env.state_card_space,
                                                                   'state_card_inf')
                    _, self.ops[i]['v_inf'], _, _, self.ops[i]['softmax'] = self._build_inference(
                        self.ops[i]['state_history_inf'], self.ops[i]['state_card_inf'], reuse=True)

            self.paramerters_place_holders = []
            parameters_op_list = []
            for parameter in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*current.*"):
                ph = tf.placeholder(tf.float32, shape=parameter.get_shape())
                self.paramerters_place_holders.append(ph)
                parameters_op_list.append(tf.assign(parameter, ph))
            self.load_op = tf.group(*parameters_op_list)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # build data pipeline
        self.data_states_card = np.zeros([2, self.flags.data_len] + self.env.state_card_space, np.int32)
        self.data_states_history = np.zeros([2, self.flags.data_len] + self.env.state_history_space, np.int32)
        self.data_actions = np.zeros([2, self.flags.data_len], np.int32)
        self.data_returns = np.zeros([2, self.flags.data_len], np.float32)
        self.data_advant = np.zeros([2, self.flags.data_len], np.float32)
        self.top = [0, 0]
        self.episode_start_index = [0, 0]

    def _build_inference(self, state_history_ph, state_card_ph, reuse=False):
        # policy and v
        state_history = tf.reshape(tf.cast(state_history_ph, tf.float32), [-1] + [reduce(lambda x, y: x * y, self.env.state_history_space)])
        state_card = tf.cast(state_card_ph, tf.float32)
        state = tf.concat([state_card, state_history], axis=1)
        with tf.variable_scope('current'):
            pi_logits, v = self._policy_v(state, reuse)
        pi_softmax = tf.nn.softmax(pi_logits, name="pi_softmax")  # for test
        with tf.variable_scope('old'):
            pi_logits_old, v_old = self._policy_v(state, reuse)
        return pi_logits, v, pi_logits_old, v_old, pi_softmax

    def _build_train(self, action, advantage, ret, pi_logits, pi_logits_old, v):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                      trainable=False)
        # build training
        # action = tf.placeholder(tf.int32, [None])
        # advantage = tf.placeholder(tf.float32, [None])
        # ret = tf.placeholder(tf.float32, [None])  # empirical return
        one_hot_actions = tf.one_hot(action, self.env.action_space, dtype=tf.float32)

        neglog_pi = tf.nn.softmax_cross_entropy_with_logits(logits=pi_logits, labels=one_hot_actions)
        log_pi = -neglog_pi
        neglog_pi_old = tf.nn.softmax_cross_entropy_with_logits(logits=pi_logits_old, labels=one_hot_actions)
        log_pi_old = -neglog_pi_old

        # entropy
        a0 = pi_logits - tf.reduce_max(pi_logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        pi_entropy = tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
        mean_pi_entropy = tf.reduce_mean(pi_entropy, name="mean_entropy")

        # kl(pi_new, pi_old)
        a1 = pi_logits_old - tf.reduce_max(pi_logits_old, axis=-1, keep_dims=True)
        ea1 = tf.exp(a1)
        z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
        kl = tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
        mean_kl = tf.reduce_mean(kl, name="mean_kl")

        ratio = tf.exp(log_pi - tf.stop_gradient(log_pi_old))  # pi_new / pi_old
        surr1 = ratio * advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - self.flags.ppo_clip, 1.0 + self.flags.ppo_clip) * advantage
        surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2), name="surrogate_loss")  # maximize surrogate objective

        vf_loss = tf.reduce_mean(tf.square(v - ret), name="value_loss")  # minimize value loss

        entropy_loss = - self.flags.encoeff * tf.reduce_mean(pi_entropy)  # maximize entropy
        # total_loss = surr_loss + vf_loss + entropy_loss
        total_loss = surr_loss + vf_loss  # not use entropy

        # self.optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        grad_var_list = optimizer.compute_gradients(total_loss)
        train_grad_var_list = []
        for i in range(len(grad_var_list)):
            if grad_var_list[i][0] is None: continue
            if grad_var_list[i][1] in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        tf.get_variable_scope().name + ".*current/linear1"):
                train_grad_var_list.append((grad_var_list[i][0]/2, grad_var_list[i][1]))
            else:
                train_grad_var_list.append(grad_var_list[i])

        apply_gradients = optimizer.apply_gradients(train_grad_var_list, global_step)
        assign_ops = []
        for (cur, old) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope=tf.get_variable_scope().name + '.*current'),
                              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope=tf.get_variable_scope().name + '.*old')):
            assign_ops.append(tf.assign(old, cur))
        copy_cur2old_op = tf.group(*assign_ops, name="copy")
        return mean_kl, mean_pi_entropy, surr_loss, vf_loss, apply_gradients, copy_cur2old_op

    def _linear_layer(self, linear_in, dim, hiddens):
        weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                  initializer=tfc.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True))
        bias = tf.get_variable('bias', [hiddens], tf.float32,
                               initializer=tf.constant_initializer(0.1))
        pre_activations = tf.add(tf.matmul(linear_in, weights), bias)
        linear_out = tf.nn.relu(pre_activations)
        return linear_out

    def _policy_v(self, state, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('linear1'):
            dim = state.get_shape().as_list()[1]; hiddens = 48
            linear1 = self._linear_layer(state, dim, hiddens)
        with tf.variable_scope('policy'):
            with tf.variable_scope('linear2'):
                dim = hiddens; hiddens = 32
                linear2 = self._linear_layer(linear1, dim, hiddens)
            with tf.variable_scope('linear3'):
                dim = hiddens; hiddens = self.env.action_space
                pi_logits = self._linear_layer(linear2, dim, hiddens)

        with tf.variable_scope('linear1'):
            tf.get_variable_scope().reuse_variables()
            dim = state.get_shape().as_list()[1]; hiddens = 48
            linear1 = self._linear_layer(state, dim, hiddens)
        with tf.variable_scope('value'):
            with tf.variable_scope('linear2'):
                dim = hiddens; hiddens = 32
                linear2 = self._linear_layer(linear1, dim, hiddens)
            with tf.variable_scope('linear3'):
                dim = hiddens; hiddens = 1
                v = self._linear_layer(linear2, dim, hiddens)
        return pi_logits, v

    def get_parameters(self):
        return self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*current.*"))

    def load_parameters(self, parameters_list):
        feed_dict = {}
        for i, j in zip(parameters_list, self.paramerters_place_holders):
            feed_dict[j] = i
        self.sess.run(self.load_op, feed_dict=feed_dict)

    def compute_self_policy(self):
        policy = [{}, {}]
        value = [{}, {}]
        for cards in XFP.possible_cards_list:
            for history in LeducRLEnv.history_string2vector:
                state_card = np.zeros([self.env.card_num], np.int32)
                state_history = LeducRLEnv.history_string2vector[history]
                pround = 1 if history in XFP.round1_states_set else 2
                if history in XFP.player1_states_set:
                    state_card[int(cards[0])] = 1
                    card = cards[0]
                    if pround == 2:
                        state_card[int(self.env.card_num) / 2 + int(cards[1])] = 1
                        card = cards[:2]
                    if card + history in policy[0]: continue
                    nn_pi_v = self.sess.run([self.ops[0]['softmax'], self.ops[0]['v_inf']],
                                            feed_dict={self.ops[0]['state_card_inf']: [state_card],
                                                       self.ops[0]['state_history_inf']: [state_history]})
                    policy[0][card + history] = nn_pi_v[0][0]
                    value[0][card + history] = nn_pi_v[1][0]
                else:
                    state_card[int(cards[2])] = 1
                    card = cards[2]
                    if pround == 2:
                        state_card[int(self.env.card_num) / 2 + int(cards[1])] = 1
                        card = cards[1:]
                    if card + history in policy[1]: continue
                    nn_pi_v = self.sess.run([self.ops[1]['softmax'], self.ops[1]['v_inf']],
                                            feed_dict={self.ops[1]['state_card_inf']: [state_card],
                                                       self.ops[1]['state_history_inf']: [state_history]})
                    policy[1][card + history] = nn_pi_v[0][0]
                    value[1][card + history] = nn_pi_v[1][0]

        self.explicit_policy = policy
        self.explicit_value = value
        return policy, value

    def collect_samples(self, num_games, cut2same_size=False):
        self.top[0] = self.top[1] = 0
        cards_indices = np.random.randint(0, len(XFP.possible_cards_list), num_games)
        for collect_for_p in range(2):
            for game in xrange(num_games):
                self.episode_start_index[collect_for_p] = self.top[collect_for_p]
                ob = self.env.reset(XFP.possible_cards_list[cards_indices[game]])
                while ob['turn'] != -1:
                    state_str = ob['card_str'] + ob['history_str']
                    position = ob['turn']
                    if position != collect_for_p:
                        action = np.random.choice(2, [], p=self.opponent_policy[position][state_str])
                    else:
                        action = np.random.choice(2, [], p=self.explicit_policy[position][state_str])
                        value = self.explicit_value[position][state_str]
                        self.data_states_card[position, self.top[position]] = ob['card']
                        self.data_states_history[position, self.top[position]] = ob['state']
                        self.data_actions[position, self.top[position]] = action
                        self.data_advant[position, self.top[position]] = value
                        self.top[position] += 1
                        assert self.top[position] < self.flags.data_len
                    ob = self.env.act(action)
                episode_return = ob['payoff'][collect_for_p]
                while self.episode_start_index[collect_for_p] != self.top[collect_for_p]:
                    self.data_returns[collect_for_p, self.episode_start_index[collect_for_p]] = episode_return
                    self.data_advant[collect_for_p, self.episode_start_index[collect_for_p]] = \
                        episode_return - self.data_advant[collect_for_p, self.episode_start_index[collect_for_p]]
                    self.episode_start_index[collect_for_p] += 1
        if cut2same_size:
            size = min(self.top)
            self.top = [size, size]

    def learn(self, policy_gradient_train_num, opponent_policy, num_games=10000):
        self.opponent_policy = opponent_policy
        for _ in range(policy_gradient_train_num):
            self.compute_self_policy()
            self.collect_samples(num_games, cut2same_size=True)
            print "iter {:d} : training from {:d} games, totally {:d} {:d} transitions".format(
                _, num_games, self.top[0], self.top[1])
            self.sess.run([self.ops[0]['copy'], self.ops[1]['copy']])

            data_pipeline_feed_dict = {}
            for i in range(2):
                data_pipeline_feed_dict[self.ops[i]['state_history']] = self.data_states_history[i, :self.top[i]]
                data_pipeline_feed_dict[self.ops[i]['state_card']] = self.data_states_card[i, :self.top[i]]
                data_pipeline_feed_dict[self.ops[i]['action']] = self.data_actions[i, :self.top[i]]
                data_pipeline_feed_dict[self.ops[i]['advantage']] = self.data_advant[i, :self.top[i]]
                data_pipeline_feed_dict[self.ops[i]['ret']] = self.data_returns[i, :self.top[i]]

            self.sess.run([self.ops[0]['iterator'].initializer, self.ops[1]['iterator'].initializer],
                          feed_dict=data_pipeline_feed_dict)

            print "    {: ^13}|{: ^13}|{: ^13}|{: ^13}|{: ^13}".format("meanKL", "meanEntropy", "surr_loss", "vfloss", "lr")
            print_frequency = 0
            while True:
                try:
                    result = self.sess.run([self.ops[0]['mean_kl'],
                                            self.ops[0]['mean_pi_entropy'],
                                            self.ops[0]['surr_loss'],
                                            self.ops[0]['vf_loss'],
                                            self.lr,
                                            self.ops[1]['mean_kl'],
                                            self.ops[1]['mean_pi_entropy'],
                                            self.ops[1]['surr_loss'],
                                            self.ops[1]['vf_loss'],
                                            self.lr,
                                            self.ops[0]['apply_gradients'],
                                            self.ops[1]['apply_gradients']])
                    if print_frequency % int(self.top[0]/self.flags.batch*self.flags.epochs/4) == 0:
                        print "p0: {: ^13.4f}|{: ^13.4f}|{: ^13.4f}|{: ^13.4f}|{: ^13.4f}".format(*result[:5])
                        print "p1: {: ^13.4f}|{: ^13.4f}|{: ^13.4f}|{: ^13.4f}|{: ^13.4f}".format(*result[5:10])
                    print_frequency += 1
                except tf.errors.OutOfRangeError:
                    break

