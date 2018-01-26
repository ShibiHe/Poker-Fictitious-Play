import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
from XFP import XFP, LeducRLEnv


class NFSP(object):
    def __init__(self, flags):
        self.flags = flags
        self.env = LeducRLEnv(card_num=flags.card_num, seed=flags.seed)
        np.random.seed(flags.seed)
        self.iter = [0, 0]
        self.epsilon = self.flags.epsilon
        self.sl_replay = [ReservoirReplay(flags, self.env), ReservoirReplay(flags, self.env)]
        self.rl_replay = [CircularReplay(flags, self.env), CircularReplay(flags, self.env)]

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
            self.ops = [{}, {}]
            for i in range(2):
                with tf.variable_scope("player" + str(i)):
                    self.ops[i]['state_history_ph'] = tf.placeholder(tf.int8, [None] + self.env.state_history_space,
                                                                     "state_history")
                    self.ops[i]['state_card_ph'] = tf.placeholder(tf.int8, [None] + self.env.state_card_space,
                                                                  "state_card")
                    self.ops[i]['state_history_ph2'] = tf.placeholder(tf.int8, [None] + self.env.state_history_space,
                                                                      "state_history2")
                    self.ops[i]['state_card_ph2'] = tf.placeholder(tf.int8, [None] + self.env.state_card_space,
                                                                   "state_card2")

                    with tf.variable_scope('current_q'):
                        self.ops[i]['q_logits_s'] = self._build_inference(self.ops[i]['state_history_ph'], self.ops[i]['state_card_ph'])

                    with tf.variable_scope('old_q'):
                        self.ops[i]['q_logits_s2_old'] = self._build_inference(self.ops[i]['state_history_ph2'], self.ops[i]['state_card_ph2'])

                    with tf.variable_scope('average_pi'):
                        self.ops[i]['pi_logits_s'] = self._build_inference(self.ops[i]['state_history_ph'], self.ops[i]['state_card_ph'])

                    assign_ops = []
                    for (cur, old) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope=tf.get_variable_scope().name + '.*current'),
                                          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope=tf.get_variable_scope().name + '.*old')):
                        assign_ops.append(tf.assign(old, cur))
                    self.ops[i]['copy'] = tf.group(*assign_ops, name="copy")

                    self.ops[i]['global_step'] = tf.get_variable("global_step", [], tf.int64,
                                                                 tf.constant_initializer(0), trainable=False)
                    self.ops[i]['action_ph'] = tf.placeholder(tf.int8, [None])
                    self.ops[i]['reward_ph'] = tf.placeholder(tf.int8, [None])
                    self.ops[i]['terminal_ph'] = tf.placeholder(tf.int8, [None])  # 1.0 is terminal
                    self.ops[i]['apply_gradients_sl'], self.ops[i]['apply_gradients_rl'], self.ops[i]['sl_loss'], self.ops[i]['rl_loss'] = \
                        self._build_train(self.ops[i]['action_ph'],
                                          self.ops[i]['pi_logits_s'],
                                          self.ops[i]['reward_ph'],
                                          self.ops[i]['terminal_ph'],
                                          self.ops[i]['q_logits_s'],
                                          self.ops[i]['q_logits_s2_old'],
                                          self.ops[i]['global_step'])

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.sess.run([self.ops[0]['copy'], self.ops[1]['copy']])

    def _build_inference(self, state_history_ph, state_card_ph, reuse=False):
        state_history = tf.reshape(tf.cast(state_history_ph, tf.float32),
                                   [-1] + [reduce(lambda x, y: x * y, self.env.state_history_space)])
        state_card = tf.cast(state_card_ph, tf.float32)
        state = tf.concat([state_card, state_history], axis=1)
        logits = self._network(state, reuse)
        return logits

    def _build_train(self, action, pi_logits_s, reward, terminal, q_logits_s, q_logits_s2_old, global_step):
        action = tf.cast(action, tf.int32)
        one_hot_actions = tf.one_hot(action, self.env.action_space, dtype=tf.float32)
        neglog_pi = tf.nn.softmax_cross_entropy_with_logits(logits=pi_logits_s, labels=one_hot_actions)

        reward = tf.cast(reward, tf.float32)
        terminal = tf.cast(terminal, tf.float32)
        target = reward + (1.0 - terminal) * tf.reduce_max(q_logits_s2_old, axis=1)
        q_s_a = tf.reduce_sum(q_logits_s * one_hot_actions, axis=1)
        loss = tf.reduce_mean(tf.square(q_s_a - tf.stop_gradient(target)))

        optimizer_sl = tf.train.GradientDescentOptimizer(self.flags.lr_sl)
        optimizer_rl = tf.train.GradientDescentOptimizer(self.flags.lr_rl)

        grad_var_list_sl = optimizer_sl.compute_gradients(neglog_pi)
        grad_var_list_rl = optimizer_rl.compute_gradients(loss)

        apply_gradients_sl = optimizer_sl.apply_gradients(grad_var_list_sl, global_step)
        apply_gradients_rl = optimizer_rl.apply_gradients(grad_var_list_rl)

        return apply_gradients_sl, apply_gradients_rl, neglog_pi, loss

    def _linear_layer(self, linear_in, dim, hiddens):
        weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                  initializer=tfc.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True))
        bias = tf.get_variable('bias', [hiddens], tf.float32,
                               initializer=tf.constant_initializer(0.1))
        pre_activations = tf.add(tf.matmul(linear_in, weights), bias)
        linear_out = tf.nn.relu(pre_activations)
        return linear_out

    def _network(self, state, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('linear1'):
            dim = state.get_shape().as_list()[1]; hiddens = 64
            linear1 = self._linear_layer(state, dim, hiddens)
        with tf.variable_scope('linear2'):
            dim = hiddens; hiddens = self.env.action_space
            logits = self._linear_layer(linear1, dim, hiddens)
        return logits

    def choose_action(self, position, state_history, state_card):
        if np.random.rand() < self.flags.anticipatory:  # epsilon greedy
            if np.random.rand() < self.epsilon:  # explore
                action = np.random.randint(0, self.env.action_space)
            else:
                q_logits_s = self.sess.run(self.ops[position]['q_logits_s'],
                                           feed_dict={self.ops[position]['state_history_ph']: [state_history],
                                                      self.ops[position]['state_card_ph']: [state_card]})
                action = np.argmax(q_logits_s[0])
            return action, 'br'  # best response
        else:
            pi_logits_s = self.sess.run(self.ops[position]['pi_logits_s'],
                                        feed_dict={self.ops[position]['state_history_ph']: [state_history],
                                                   self.ops[position]['state_card_ph']: [state_card]})
            prob = np.exp(pi_logits_s[0])
            prob = prob / np.sum(prob)
            action = np.random.choice(self.env.action_space, p=prob)
            return action, 'avg'  # average

    def play_game(self):
        ob = self.env.reset()
        while True:
            position = ob['turn']
            if ob['turn'] == -1:
                self.rl_replay[0].add_terminal(ob['payoff'][0])
                self.rl_replay[1].add_terminal(ob['payoff'][1])
                break
            state_history = ob['state']
            state_card = ob['card']
            action, tag = self.choose_action(position, state_history, state_card)
            if tag == 'br':  # greedy best response
                self.sl_replay[position].add(state_history, state_card, action)
            self.rl_replay[position].add(state_history, state_card, action, 0, False)
            self.iter[position] += 1
            if self.iter[position] % self.flags.train_frequency == 0 and self.iter[position] > self.flags.train_start:
                self.train(position)

            ob = self.env.act(action)

    def train(self, position):
        batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer = self.sl_replay[position].get_random_batch()
        global_step, _ = self.sess.run([self.ops[position]['global_step'], self.ops[position]['apply_gradients_sl']],
                                       feed_dict={self.ops[position]['state_history_ph']: batch_state_history_buffer,
                                                  self.ops[position]['state_card_ph']: batch_state_card_buffer,
                                                  self.ops[position]['action_ph']: batch_action_buffer})
        batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer, \
            batch_reward_buffer, batch_terminal_buffer, batch_state_history_buffer2, \
            batch_state_card_buffer2 = self.rl_replay[position].get_random_batch()
        _ = self.sess.run(self.ops[position]['apply_gradients_rl'],
                          feed_dict={self.ops[position]['state_history_ph']: batch_state_history_buffer,
                                     self.ops[position]['state_card_ph']: batch_state_card_buffer,
                                     self.ops[position]['action_ph']: batch_action_buffer,
                                     self.ops[position]['reward_ph']: batch_reward_buffer,
                                     self.ops[position]['terminal_ph']: batch_terminal_buffer,
                                     self.ops[position]['state_history_ph2']: batch_state_history_buffer2,
                                     self.ops[position]['state_card_ph2']: batch_state_card_buffer2})
        if global_step % self.flags.refit == 0:
            self.sess.run(self.ops[position]['copy'])
        print 'train policy {:d} at step {:d}, global_step={:d}, epsilon={:.4f}'.format(position, self.iter[position], global_step, self.epsilon)

    def compute_self_policy(self):
        policy = [{}, {}]
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
                    q_p, pi = self.sess.run([self.ops[0]['q_logits_s'], self.ops[0]['pi_logits_s']],
                                            feed_dict={self.ops[0]['state_history_ph']: [state_history],
                                                       self.ops[0]['state_card_ph']: [state_card]})
                    q_max_a = np.argmax(q_p[0])
                    prob = np.exp(pi[0])
                    prob = prob / np.sum(prob)
                    prob = prob * (1 - self.flags.anticipatory)
                    prob[q_max_a] += self.flags.anticipatory * 1.0
                    policy[0][card + history] = prob
                else:
                    state_card[int(cards[2])] = 1
                    card = cards[2]
                    if pround == 2:
                        state_card[int(self.env.card_num) / 2 + int(cards[1])] = 1
                        card = cards[1:]
                    if card + history in policy[1]: continue
                    q_p, pi = self.sess.run([self.ops[1]['q_logits_s'], self.ops[1]['pi_logits_s']],
                                            feed_dict={self.ops[1]['state_history_ph']: [state_history],
                                                       self.ops[1]['state_card_ph']: [state_card]})
                    q_max_a = np.argmax(q_p[0])
                    prob = np.exp(pi[0])
                    prob = prob / np.sum(prob)
                    prob = prob * (1 - self.flags.anticipatory)
                    prob[q_max_a] += self.flags.anticipatory * 1.0
                    policy[1][card + history] = prob

        return policy


class ReservoirReplay(object):  # sl
    def __init__(self, flags, env):
        self.flags = flags
        self.env = env
        self.state_history_buffer = np.zeros([self.flags.sl_len] + self.env.state_history_space, np.int8)
        self.state_card_buffer = np.zeros([self.flags.sl_len] + self.env.state_card_space, np.int8)
        self.action_buffer = np.zeros([self.flags.sl_len], np.int8)
        self.size = 0
        self.top = 0

    def add(self, state_history, state_card, action):
        if self.size < self.flags.sl_len:
            self.state_history_buffer[self.top] = state_history
            self.state_card_buffer[self.top] = state_card
            self.action_buffer[self.top] = action
            self.top += 1
            self.size += 1
        else:
            prob_add = float(self.flags.sl_len) / float(self.top + 1)
            if np.random.rand() < prob_add:
                index = np.random.randint(0, self.flags.sl_len)
                self.state_history_buffer[index] = state_history
                self.state_card_buffer[index] = state_card
                self.action_buffer[index] = action
            self.top += 1

    def get_random_batch(self):
        indices = np.random.randint(0, self.size, self.flags.batch)
        batch_state_history_buffer = np.take(self.state_history_buffer, indices, axis=0)
        batch_state_card_buffer = np.take(self.state_card_buffer, indices, axis=0)
        batch_action_buffer = np.take(self.action_buffer, indices)
        return batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer


class CircularReplay(object):  # rl
    def __init__(self, flags, env):
        self.flags = flags
        self.env = env
        self.state_history_buffer = np.zeros([self.flags.rl_len] + self.env.state_history_space, np.int8)
        self.state_card_buffer = np.zeros([self.flags.rl_len] + self.env.state_card_space, np.int8)
        self.action_buffer = np.zeros([self.flags.rl_len], np.int8)
        self.reward_buffer = np.zeros([self.flags.rl_len], np.int8)
        self.terminal_buffer = np.zeros([self.flags.rl_len], np.int8)
        self.size = 0
        self.top = 0
        self.bottom = 0

    def add(self, state_history, state_card, action, reward, terminal):
        self.state_history_buffer[self.top] = state_history
        self.state_card_buffer[self.top] = state_card
        self.action_buffer[self.top] = action
        self.reward_buffer[self.top] = reward
        self.terminal_buffer[self.top] = terminal
        if self.size == self.flags.rl_len:
            self.bottom = (self.bottom + 1) % self.flags.rl_len
        else:
            self.size += 1
        self.top = (self.top + 1) % self.flags.rl_len

    def add_terminal(self, reward):
        last_top = (self.top - 1) % self.flags.rl_len
        self.reward_buffer[last_top] = reward
        self.terminal_buffer[last_top] = True

    def get_random_batch(self):
        indices = np.random.randint(0, self.size, self.flags.batch)
        indices2 = indices + 1
        batch_state_history_buffer = np.take(self.state_history_buffer, indices, axis=0)
        batch_state_card_buffer = np.take(self.state_card_buffer, indices, axis=0)
        batch_state_history_buffer2 = np.take(self.state_history_buffer, indices2, axis=0, mode='wrap')
        batch_state_card_buffer2 = np.take(self.state_card_buffer, indices2, axis=0, mode='wrap')
        batch_action_buffer = np.take(self.action_buffer, indices)
        batch_reward_buffer = np.take(self.reward_buffer, indices)
        batch_terminal_buffer = np.take(self.terminal_buffer, indices)
        return batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer, batch_reward_buffer, \
            batch_terminal_buffer, batch_state_history_buffer2, batch_state_card_buffer2


