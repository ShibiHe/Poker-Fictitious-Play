import numpy as np
import tensorflow as tf
import pickle
import copy
from XFP import XFP, LeducRLEnv
from PPOFP import PPO
FLAGS = tf.app.flags.FLAGS

# Experiment settings
tf.app.flags.DEFINE_integer('seed', 100, 'random seed')
tf.app.flags.DEFINE_integer('num_cpu', 8, 'num of cpus')
tf.app.flags.DEFINE_integer('card_num', 6, 'card numbers in leduc')
tf.app.flags.DEFINE_integer('data_len', 3*10000, 'data set length')
tf.app.flags.DEFINE_integer('epochs', 4, 'training epochs')
tf.app.flags.DEFINE_integer('batch', 256, 'batch size')
tf.app.flags.DEFINE_float('ppo_clip', 0.2, 'PPO clip parameter')
tf.app.flags.DEFINE_float('encoeff', 0.0, 'entropy bonus')
tf.app.flags.DEFINE_float('lr', 0.005, 'learning rate')
tf.app.flags.DEFINE_integer('fsp_iter', 100, 'fictitious play iterations')
tf.app.flags.DEFINE_bool("use_gpu", False, "run on GPU")

xfp = XFP(card_num=FLAGS.card_num, seed=FLAGS.seed)


def evaluate_exploitabality(p1, p2):
    xfp.opponent_policy_p2 = p2
    xfp.compute_p1_best_response()
    br1_op = XFP.convert_q_s_a2greedy_policy(xfp.q_value1_final)
    xfp.opponent_policy_p1 = p1
    xfp.compute_p2_best_response()
    br2_op = XFP.convert_q_s_a2greedy_policy(xfp.q_value2_final)

    # compute optimal best response payoff
    realization_br1_op = XFP.compute_realization(br1_op, p2)
    realization_br2_op = XFP.compute_realization(p1, br2_op)
    e = [XFP.compute_payoff_given_realization(realization_br1_op)[0],
         XFP.compute_payoff_given_realization(realization_br2_op)[1]]
    exploitability = (e[0] + e[1]) / 2.0
    xfp.finish()
    return exploitability, e


def evaluatate_best_response(iteration, p1, p2, br1, br2):
    realization_old = XFP.compute_realization(p1, p2)
    realization_rl_br1 = XFP.compute_realization(br1, p2)
    realization_rl_br2 = XFP.compute_realization(p1, br2)
    mix_realization_p1 = XFP.mix_realization(realization_rl_br1, realization_old, 0.1)
    mix_realization_p2 = XFP.mix_realization(realization_rl_br2, realization_old, 0.1)
    new_p1, _ = XFP.compute_realization2policy(mix_realization_p1)
    _, new_p2 = XFP.compute_realization2policy(mix_realization_p2)
    exploitability, _ = evaluate_exploitabality(new_p1, new_p2)
    return exploitability, new_p1, new_p2


def train():
    agent = PPO(FLAGS)
    p1, p2 = xfp.get_uniform_policy()
    iteration = 0
    while iteration < FLAGS.fsp_iter:
        print '---------------Fictitious Self-play Iteration', iteration, '------------------'
        exploitability, e = evaluate_exploitabality(p1, p2)
        print "optimal best response policy for p1 achieves {:.4f}\n" \
              "optimal best response policy for p2 achieves {:.4f}, exploitability is {:.4f}".format(
            e[0], e[1], exploitability)
        agent.learn(10, [p1, p2])
        br_policy, _ = agent.compute_self_policy()
        mixture_e, p1_new, p2_new = evaluatate_best_response(iteration, p1, p2, br_policy[0], br_policy[1])
        print mixture_e

        iteration += 1
        p1 = p1_new
        p2 = p2_new


def some_tests():
    flags = copy.deepcopy(FLAGS)
    flags.batch = 4
    flags.epochs = 1
    agent = PPO(flags)
    # test compute policy
    agent.compute_self_policy()
    for key, item in agent.explicit_policy[0].iteritems():
        state_card = np.zeros([agent.env.card_num], np.int32)
        card, state = xfp.get_card_state_state(key)
        state_card[int(card[0])] = 1
        if len(card) == 2:
            state_card[int(agent.env.card_num) / 2 + int(card[1])] = 1
        state_history = agent.env.history_string2vector[state]
        res = agent.sess.run(agent.ops[0]['softmax'],
                             feed_dict={agent.ops[0]['state_card_inf']: [state_card],
                                        agent.ops[0]['state_history_inf']: [state_history]})
        assert res[0][0] == item[0] and res[0][1] == item[1]
    for key, item in agent.explicit_policy[1].iteritems():
        state_card = np.zeros([agent.env.card_num], np.int32)
        card, state = xfp.get_card_state_state(key)
        if len(card) == 2:
            state_card[int(card[1])] = 1
            state_card[int(agent.env.card_num) / 2 + int(card[0])] = 1
        else:
            state_card[int(card[0])] = 1
        state_history = agent.env.history_string2vector[state]
        res = agent.sess.run(agent.ops[1]['softmax'],
                             feed_dict={agent.ops[1]['state_card_inf']: [state_card],
                                        agent.ops[1]['state_history_inf']: [state_history]})
        assert res[0][0] == item[0] and res[0][1] == item[1]

    # test load parameters
    parameters = agent.get_parameters()
    agent.sess.run(agent.init)
    agent.load_parameters(parameters)
    for i, j in zip(parameters, agent.get_parameters()):
        assert np.allclose(i, j)

    # test collect_sample and data pipeline
    agent.opponent_policy = agent.explicit_policy
    agent.collect_samples(100)
    # print agent.top
    data = [{}, {}]
    for i in agent.data_returns[0, :agent.top[0]]:
        data[0][i] = data[0][i] + 1 if i in data[0] else 1
    for i in agent.data_returns[1, :agent.top[1]]:
        data[1][i] = data[1][i] + 1 if i in data[1] else 1
    data2 = [{}, {}]
    for i in range(2):
        agent.sess.run(agent.ops[i]['iterator'].initializer,
                       feed_dict={agent.ops[i]['state_history']: agent.data_states_history[i, :agent.top[i]],
                                  agent.ops[i]['state_card']: agent.data_states_card[i, :agent.top[i]],
                                  agent.ops[i]['action']: agent.data_actions[i, :agent.top[i]],
                                  agent.ops[i]['advantage']: agent.data_advant[i, :agent.top[i]],
                                  agent.ops[i]['ret']: agent.data_returns[i, :agent.top[i]]})
        while True:
            try:
                x = agent.sess.run(agent.ops[i]['next_element'])
                for j in x[4]:
                    data2[i][j] = data2[i][j] + 1 if j in data2[i] else 1
            except tf.errors.OutOfRangeError:
                break
    assert data2 == data

    # test learn
    p1, p2 = xfp.get_random_policy()
    agent.learn(10, [p1, p2], 100)


if __name__ == '__main__':
    train()
    # some_tests()


