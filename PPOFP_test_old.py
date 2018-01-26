import numpy as np
import tensorflow as tf
import pickle
import copy
from XFP import XFP, LeducRLEnv
from PPOFP_old import PPO
FLAGS = tf.app.flags.FLAGS

# Experiment settings
tf.app.flags.DEFINE_integer('seed', 102, 'random seed')
tf.app.flags.DEFINE_integer('num_cpu', 8, 'num of cpus')
tf.app.flags.DEFINE_integer('card_num', 6, 'card numbers in leduc')
tf.app.flags.DEFINE_integer('data_len', 3*10000, 'data set length')
tf.app.flags.DEFINE_integer('epochs', 4, 'training epochs')
tf.app.flags.DEFINE_integer('batch', 256, 'batch size')
tf.app.flags.DEFINE_float('ppo_clip', 0.2, 'PPO clip parameter')
tf.app.flags.DEFINE_float('encoeff', 0.0, 'entropy bonus')
tf.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
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


def train(iter_start=None, p1=None, p2=None):
    agent0 = PPO(FLAGS, 0)
    agent1 = PPO(FLAGS, 1)
    if iter_start is None:
        p1, p2 = xfp.get_random_policy()
        iteration = 0
    else:
        iteration = iter_start
    while iteration < FLAGS.fsp_iter:
        print '---------------Fictitious Self-play Iteration', iteration, '------------------'
        # evaluate optimal best response for p1 p2
        exploitability, e = evaluate_exploitabality(p1, p2)
        print "optimal best response policy for p1 achieves {:.4f}\n" \
              "optimal best response policy for p2 achieves {:.4f}, exploitability is {:.4f}".format(
            e[0], e[1], exploitability)

        # policy gradient
        # run different init, keep the best
        p1_br_over_inits = None
        p1_br_payoff_over_inits = -1000
        p2_br_over_inits = None
        p2_br_payoff_over_inits = -1000
        for seed_times in range(1):
            agent0.sess.run(agent0.init)
            agent1.sess.run(agent1.init)
            p1_best_behaved_br, p1_best_behaved_br_payoff = agent0.learn(15, 10000, p2)
            p2_best_behaved_br, p2_best_behaved_br_payoff = agent1.learn(15, 10000, p1)
            if p1_best_behaved_br_payoff > p1_br_payoff_over_inits:
                p1_br_payoff_over_inits = p1_best_behaved_br_payoff
                p1_br_over_inits = p1_best_behaved_br
            if p2_best_behaved_br_payoff > p2_br_payoff_over_inits:
                p2_br_payoff_over_inits = p2_best_behaved_br_payoff
                p2_br_over_inits = p2_best_behaved_br

        # do mixture
        mixture_e, p1_new, p2_new = evaluatate_best_response(iteration, p1, p2, p1_br_over_inits, p2_br_over_inits)
        import os
        if not os.path.exists("{:d}".format(FLAGS.seed)):
            os.makedirs("{:d}".format(FLAGS.seed))
        with open('{:d}/iter{:d}_{:.2f}_{:.2f}_{:.2f}_{:.2}_{:.2}.pickle'.format(
                FLAGS.seed, iteration, mixture_e, e[0], e[1],
                p1_best_behaved_br_payoff, p2_best_behaved_br_payoff), 'wb') as handle:
            pickle.dump([p1, p2, p1_br_over_inits, p2_br_over_inits], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print "best response payoff: p1 {:.4f}  p2 {:.4f}".format(p1_best_behaved_br_payoff, p2_best_behaved_br_payoff)
        p1 = p1_new
        p2 = p2_new
        iteration += 1


def train_from_pickle(filename, iteration=None):
    iteration = int(filename[4:filename.find('_')]) if iteration is None else iteration
    with open(filename, 'rb') as handle:
        p1, p2, p1_best_behaved_br, p2_best_behaved_br = pickle.load(handle)
    e, p1, p2 = evaluatate_best_response(iteration, p1, p2, p1_best_behaved_br, p2_best_behaved_br)
    print e
    train(iter_start=iteration+1, p1=p1, p2=p2)


if __name__ == '__main__':
    train()