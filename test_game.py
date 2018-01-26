import numpy as np
from XFP import XFP, LeducRLEnv

seed = 100
np.random.seed(seed)
env = XFP()
env.compute_p2_best_response()
env.compute_p1_best_response()

p1 = 0.0
p2 = 0.0

for i in range(100):
    cards = env.possible_cards_list[np.random.randint(24)]
    game_state = ""
    while True:
        if game_state in env.ending:
            v1, v2 = env.compute_payoff(cards, game_state)
            p1 += v1
            p2 += v2
            # print cards, game_state, v1, v2
            break
        if game_state in env.round1_states_set:
            rd = 1
        else:
            rd = 2
        if game_state in env.player1_states_set:
            # action = env.choose_action_p1(game_state, cards, rd)
            if np.random.randint(0, 100) < 37:
                action = 'C'
            else:
                action = 'B'
            game_state = game_state + action
        else:
            action = env.choose_action_p2(game_state, cards, rd)
            # if np.random.randint(0, 100) < 37:
            #     action = 'C'
            # else:
            #     action = 'B'
            game_state = game_state + action

print p1, p2
assert seed != 100 or p2 == 60 and seed == 100

# test realization
policy_p1 = {}
policy_p2 = {}

for key, _ in env.q_value1_final.iteritems():
    p = np.random.rand()
    policy_p1[key] = [p, 1.0 - p]

for key, _ in env.q_value2_final.iteritems():
    p = np.random.rand()
    policy_p2[key] = [p, 1.0 - p]


realization = XFP.compute_realization(policy_p1, policy_p2)

print XFP.tournament(seed, 10000, policy_p1, policy_p2)
print XFP.compute_payoff_given_realization(realization)
p1, p2 = XFP.compute_realization2policy(realization)

for key in policy_p1:
    assert np.allclose(policy_p1[key], p1[key])
for key in policy_p2:
    assert np.allclose(policy_p2[key], p2[key])

# test mix policy
env.finish()
env.opponent_policy_p2 = policy_p2
env.opponent_policy_p1 = policy_p1
env.compute_p1_best_response()
env.compute_p2_best_response()

player1_best_response_policy = XFP.convert_q_s_a2greedy_policy(env.q_value1_final)
player2_best_response_policy = XFP.convert_q_s_a2greedy_policy(env.q_value2_final)
env.finish()

# test best response with realization
env.opponent_realization_p1 = realization
env.opponent_realization_p2 = realization
env.opponent_realization_enable = True
env.compute_p1_best_response()
env.compute_p2_best_response()
player1_best_response_policy_given_realization = XFP.convert_q_s_a2greedy_policy(env.q_value1_final)
player2_best_response_policy_given_realization = XFP.convert_q_s_a2greedy_policy(env.q_value2_final)
for key, item in player1_best_response_policy.iteritems():
    assert np.allclose(item, player1_best_response_policy_given_realization[key])
for key, item in player2_best_response_policy.iteritems():
    assert np.allclose(item, player2_best_response_policy_given_realization[key])
env.finish()
env.opponent_realization_enable = False

print XFP.tournament(seed, 1000, player1_best_response_policy, policy_p2)
print XFP.tournament(seed, 1000, policy_p1, player2_best_response_policy)

realization_old = XFP.compute_realization(policy_p1, policy_p2)
realization_br1 = XFP.compute_realization(player1_best_response_policy, policy_p2)
realization_br2 = XFP.compute_realization(policy_p1, player2_best_response_policy)

mix_realization_p1 = XFP.mix_realization(realization_br1, realization_old, 0.5)
mix_realization_p2 = XFP.mix_realization(realization_br2, realization_old, 0.5)

new_policy_p1, p2_extra = XFP.compute_realization2policy(mix_realization_p1)
p1_extra, new_policy_p2 = XFP.compute_realization2policy(mix_realization_p2)

for key in policy_p1:
    assert np.allclose(policy_p1[key], p1_extra[key])
for key in policy_p2:
    assert np.allclose(policy_p2[key], p2_extra[key])

print XFP.tournament(seed, 1000, new_policy_p1, policy_p2)
print XFP.tournament(seed, 1000, policy_p1, new_policy_p2)

# test LeducRLEnv

leduc = LeducRLEnv(seed=seed)
print leduc.reset()
print leduc.act(1)
print leduc.act(1)
print leduc.act(1)
print leduc.act(1)

