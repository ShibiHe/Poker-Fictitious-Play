import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
from XFP import XFP


def visualize(thing, realization=False):
    for item in sorted(list(thing.iteritems()), key=lambda x: len(x[0])):
        if item[0] in XFP.ending:
            continue
        if realization:
            print item[0], [item[1]['C'][0]/(item[1]['C'][0] + item[1]['B'][0]), item[1]['B'][0]/(item[1]['C'][0] + item[1]['B'][0])]
        else:
            print item[0], item[1]

seed = 100
tot_iter = 100
tournament_games = 1000
enable_tournament = True
np.random.seed(seed)
env = XFP(card_num=6)
policy_p1, policy_p2 = env.get_random_policy()


exploitability_performance = np.zeros([tot_iter])


def fictitious_play(env, policy_p1, policy_p2):
    for iteration in range(tot_iter):
        print '---------------iteration', iteration, '------------------'
        env.opponent_policy_p1 = policy_p1
        env.opponent_policy_p2 = policy_p2
        env.compute_p1_best_response()
        env.compute_p2_best_response()
        player1_best_response_policy = XFP.convert_q_s_a2greedy_policy(env.q_value1_final)
        player2_best_response_policy = XFP.convert_q_s_a2greedy_policy(env.q_value2_final)

        realization_old = XFP.compute_realization(policy_p1, policy_p2)
        realization_br1 = XFP.compute_realization(player1_best_response_policy, policy_p2)
        realization_br2 = XFP.compute_realization(policy_p1, player2_best_response_policy)

        e = [XFP.compute_payoff_given_realization(realization_br1)[0],
             XFP.compute_payoff_given_realization(realization_br2)[1]]
        exploitability_performance[iteration] = (e[0] + e[1]) / 2.0
        print 'exploitability=', e, exploitability_performance[iteration]

        mix_realization_p1 = XFP.mix_realization(realization_br1, realization_old, 1.0 / (iteration + 2.0))
        mix_realization_p2 = XFP.mix_realization(realization_br2, realization_old, 1.0 / (iteration + 2.0))

        policy_p1, _ = XFP.compute_realization2policy(mix_realization_p1)
        _, policy_p2 = XFP.compute_realization2policy(mix_realization_p2)
        env.finish()


def fictitious_play_realization(env, realization):  # much faster
    env.opponent_realization_enable = True
    for iteration in range(tot_iter):
        print '---------------iteration', iteration, '------------------'
        env.opponent_realization_p1 = realization
        env.opponent_realization_p2 = realization
        env.compute_p1_best_response()
        env.compute_p2_best_response()
        player1_best_response_policy = XFP.convert_q_s_a2greedy_policy(env.q_value1_final)
        player2_best_response_policy = XFP.convert_q_s_a2greedy_policy(env.q_value2_final)

        realization_br1 = XFP.compute_realization(player1_best_response_policy, realization)
        realization_br2 = XFP.compute_realization(realization, player2_best_response_policy)

        e = [XFP.compute_payoff_given_realization(realization_br1)[0],
             XFP.compute_payoff_given_realization(realization_br2)[1]]
        exploitability_performance[iteration] = (e[0] + e[1]) / 2.0
        print 'exploitability=', e, exploitability_performance[iteration]
        if enable_tournament:
            print 'tournament result:', \
                XFP.tournament(seed, tournament_games, player1_best_response_policy, realization), \
                XFP.tournament(seed, tournament_games, realization, player2_best_response_policy)

        mix_realization_p1 = XFP.mix_realization(realization_br1, realization, 0.1)
        mix_realization_p2 = XFP.mix_realization(realization_br2, realization, 0.1)

        realization = XFP.compute_realization(mix_realization_p1, mix_realization_p2)
        env.finish()
    env.opponent_realization_enable = False
    return realization


# fictitious_play(env, policy_p1, policy_p2)

realization_start = XFP.compute_realization(policy_p1, policy_p2)
realization_end = fictitious_play_realization(env, realization_start)

# plt.figure()
# plt.plot(np.arange(1, 2*tot_iter+1, 2), exploitability_performance, 'o-')
# plt.title('Leduc Holdem')
# plt.ylabel('exploitability')
# plt.xlabel('iterations')
# plt.show()

# plt.figure()
# xs = np.logspace(0, 3, 20000)
# ys = exploitability_performance[(xs-1.0).astype(np.int)]
# print xs, ys
# axis = plt.gca()
# axis.set_xscale('log')
# plt.plot(xs, ys, '-')
# plt.show()

