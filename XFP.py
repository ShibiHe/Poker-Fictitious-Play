import numpy as np


class XFP(object):
    possible_cards_list = None
    ending = None
    player1_states_set = None
    player2_states_set = None
    round1_states_set = None
    bool_init = False

    def __init__(self, verbose=False, card_num=6, seed=None):
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)
        self.player1_states_set = set()
        self.player2_states_set = set()
        self.round1_states_set = set()

        self.ending = {}
        # build the tree
        self.dfs(1, 0, "", 1, 1)

        XFP.ending = self.ending
        XFP.player1_states_set = self.player1_states_set
        XFP.player2_states_set = self.player2_states_set
        XFP.round1_states_set = self.round1_states_set

        if verbose:
            print sorted(self.player1_states_set, key=lambda x: len(x))
            print sorted(self.player2_states_set, key=lambda x: len(x))
            print sorted(self.round1_states_set, key=lambda x: len(x))

        self.possible_cards = set()
        for i in range(card_num):
            for j in range(card_num):
                for k in range(card_num):
                    if i == j or j == k or i == k:
                        continue
                    self.possible_cards.add(str(i % (card_num >> 1)) + str(j % (card_num >> 1)) + str(k % (card_num >> 1)))
        if verbose:
            print len(self.possible_cards), sorted(self.possible_cards)
            print 'ending=', XFP.ending.keys()
        XFP.possible_cards_list = list(self.possible_cards)

        self.q_value1 = {}
        self.q_value2 = {}
        self.q_value1_final = {}
        self.q_value2_final = {}

        self.opponent_policy_p1 = None
        self.opponent_policy_p2 = None
        self.opponent_realization_enable = False
        self.opponent_realization_p1 = None
        self.opponent_realization_p2 = None

        XFP.bool_init = True

    def finish(self):
        self.q_value1 = {}
        self.q_value2 = {}
        self.q_value1_final = {}
        self.q_value2_final = {}

        self.opponent_policy_p1 = None
        self.opponent_policy_p2 = None
        self.opponent_realization_p1 = None
        self.opponent_realization_p2 = None

    @staticmethod
    def compute_payoff(card_state, ending_state):
        inv1, inv2 = XFP.ending[ending_state]
        if inv1 == inv2:
            if card_state[0] == card_state[1]:
                return inv1, -inv2
            if card_state[1] == card_state[2]:
                return -inv1, inv2
            if card_state[0] > card_state[2]:
                return inv1, -inv2
            if card_state[0] < card_state[2]:
                return -inv1, inv2
            return 0.0, 0.0
        if inv1 < inv2:
            return -inv1, inv1
        return inv2, -inv2

    @staticmethod
    def get_q_s_a(q, card_state, state, action):
        if card_state + state in q:
            q_s_a = q[card_state + state]
            if action in q_s_a:
                return q_s_a[action]
        return None

    @staticmethod
    def update_q_s_a(q, card_state, state, action, value):
        if card_state + state in q:
            q_s_a = q[card_state + state]
            q_s_a[action] = value
        else:
            q[card_state + state] = {action: value}

    def dynamic_dfs_p1(self, card_state, state, action):
        if self.get_q_s_a(self.q_value1, card_state, state, action) is not None:
            return self.get_q_s_a(self.q_value1, card_state, state, action)

        if state in self.ending:
            value, _ = self.compute_payoff(card_state, state)
            self.update_q_s_a(self.q_value1, card_state, state, action, value)
            return value
        else:
            state_next = state + action
            if state_next in self.ending:
                value, _ = self.compute_payoff(card_state, state_next)
            elif state_next in self.player1_states_set:
                value = max(self.dynamic_dfs_p1(card_state, state_next, 'C'), self.dynamic_dfs_p1(card_state, state_next, 'B'))
            elif state_next in self.player2_states_set:
                if state_next in self.round1_states_set:
                    rd = 1
                else:
                    rd = 2
                prob = self.compute_opponent_policy(1, rd, state_next, card_state)
                value = prob[0] * max(self.dynamic_dfs_p1(card_state, state_next + 'C', 'C'),
                                      self.dynamic_dfs_p1(card_state, state_next + 'C', 'B'))
                value += prob[1] * max(self.dynamic_dfs_p1(card_state, state_next + 'B', 'C'),
                                       self.dynamic_dfs_p1(card_state, state_next + 'B', 'B'))
            else:
                assert False
        self.update_q_s_a(self.q_value1, card_state, state, action, value)

        if state in XFP.player1_states_set:
            if state in XFP.round1_states_set:
                card = card_state[0]
            else:
                card = card_state[:2]
            if self.get_q_s_a(self.q_value1_final, card, state, action) is not None:
                item = self.get_q_s_a(self.q_value1_final, card, state, action)
            else:
                item = [0.0, 0.0]
            item = [item[0] + value, item[1] + 1.0]
            self.update_q_s_a(self.q_value1_final, card, state, action, item)
        return value

    def dynamic_dfs_p2(self, card_state, state, action):
        if self.get_q_s_a(self.q_value2, card_state, state, action) is not None:
            return self.get_q_s_a(self.q_value2, card_state, state, action)

        if state in self.ending:
            _, value = self.compute_payoff(card_state, state)
            self.update_q_s_a(self.q_value2, card_state, state, action, value)
            return value
        else:
            state_next = state + action
            if state_next in self.ending:
                _, value = self.compute_payoff(card_state, state_next)
            elif state_next in self.player2_states_set:
                value = max(self.dynamic_dfs_p2(card_state, state_next, 'C'), self.dynamic_dfs_p2(card_state, state_next, 'B'))
            elif state_next in self.player1_states_set:
                if state_next in self.round1_states_set:
                    rd = 1
                else:
                    rd = 2
                prob = self.compute_opponent_policy(2, rd, state_next, card_state)
                value = prob[0] * max(self.dynamic_dfs_p2(card_state, state_next + 'C', 'C'),
                                      self.dynamic_dfs_p2(card_state, state_next + 'C', 'B'))
                value += prob[1] * max(self.dynamic_dfs_p2(card_state, state_next + 'B', 'C'),
                                       self.dynamic_dfs_p2(card_state, state_next + 'B', 'B'))
            else:
                assert False
        self.update_q_s_a(self.q_value2, card_state, state, action, value)

        if state in XFP.player2_states_set:
            if state in XFP.round1_states_set:
                card = card_state[2]
            else:
                card = card_state[1:]
            if self.get_q_s_a(self.q_value2_final, card, state, action) is not None:
                item = self.get_q_s_a(self.q_value2_final, card, state, action)
            else:
                item = [0.0, 0.0]
            item = [item[0] + value, item[1] + 1.0]
            self.update_q_s_a(self.q_value2_final, card, state, action, item)
        return value

    def get_random_policy(self):
        if len(self.q_value1_final) == 0:
            self.compute_p1_best_response()
        if len(self.q_value2_final) == 0:
            self.compute_p2_best_response()
        policy_p1 = {}
        policy_p2 = {}

        for key, _ in self.q_value1_final.iteritems():
            p = np.random.rand()
            policy_p1[key] = [p, 1.0 - p]

        for key, _ in self.q_value2_final.iteritems():
            p = np.random.rand()
            policy_p2[key] = [p, 1.0 - p]
        return policy_p1, policy_p2

    def get_uniform_policy(self):
        if len(self.q_value1_final) == 0:
            self.compute_p1_best_response()
        if len(self.q_value2_final) == 0:
            self.compute_p2_best_response()
        policy_p1 = {}
        policy_p2 = {}

        for key, _ in self.q_value1_final.iteritems():
            policy_p1[key] = [0.5, 0.5]

        for key, _ in self.q_value2_final.iteritems():
            policy_p2[key] = [0.5, 0.5]
        return policy_p1, policy_p2

    def compute_opponent_policy(self, player, rd, opponent_state, card_state):
        #  use policy
        if not self.opponent_realization_enable:
            if player == 1:
                if self.opponent_policy_p2 is None:
                    return 0.37, 0.63
                if rd == 1:
                    return self.opponent_policy_p2[card_state[2] + opponent_state]
                if rd == 2:
                    return self.opponent_policy_p2[card_state[1:] + opponent_state]
            else:
                if self.opponent_policy_p1 is None:
                    return 0.37, 0.63
                if rd == 1:
                    return self.opponent_policy_p1[card_state[0] + opponent_state]
                if rd == 2:
                    return self.opponent_policy_p1[card_state[:2] + opponent_state]
        #  use realization
        else:
            if player == 1:
                if self.opponent_realization_p2 is None:
                    return 0.37, 0.63
                if rd == 1:
                    v1 = self.opponent_realization_p2[card_state[2] + opponent_state]['C'][0]
                    v2 = self.opponent_realization_p2[card_state[2] + opponent_state]['B'][0]
                if rd == 2:
                    v1 = self.opponent_realization_p2[card_state[1:] + opponent_state]['C'][0]
                    v2 = self.opponent_realization_p2[card_state[1:] + opponent_state]['B'][0]
            else:
                if self.opponent_realization_p1 is None:
                    return 0.37, 0.63
                if rd == 1:
                    v1 = self.opponent_realization_p1[card_state[0] + opponent_state]['C'][0]
                    v2 = self.opponent_realization_p1[card_state[0] + opponent_state]['B'][0]
                if rd == 2:
                    v1 = self.opponent_realization_p1[card_state[:2] + opponent_state]['C'][0]
                    v2 = self.opponent_realization_p1[card_state[:2] + opponent_state]['B'][0]
            return [v1 / (v1 + v2), v2 / (v1 + v2)]

    def compute_p1_best_response(self):
        for cards in self.possible_cards_list:
            for state in self.player1_states_set:
                self.dynamic_dfs_p1(cards, state, 'C')
                self.dynamic_dfs_p1(cards, state, 'B')

    def compute_p2_best_response(self):
        for cards in self.possible_cards_list:
            for state in self.player2_states_set:
                self.dynamic_dfs_p2(cards, state, 'C')
                self.dynamic_dfs_p2(cards, state, 'B')

    def choose_action_p1(self, state, incomplete_card, pround):
        if pround == 1:
            vc, _ = self.get_q_s_a(self.q_value1_final, incomplete_card[0], state, 'C')
            vb, _ = self.get_q_s_a(self.q_value1_final, incomplete_card[0], state, 'B')
            return 'C' if vc > vb else 'B'
        if pround == 2:
            vc, _ = self.get_q_s_a(self.q_value1_final, incomplete_card[:2], state, 'C')
            vb, _ = self.get_q_s_a(self.q_value1_final, incomplete_card[:2], state, 'B')
            return 'C' if vc > vb else 'B'

    def choose_action_p2(self, state, incomplete_card, pround):
        if pround == 1:
            vc = self.get_q_s_a(self.q_value2_final, incomplete_card[2], state, 'C')
            vb = self.get_q_s_a(self.q_value2_final, incomplete_card[2], state, 'B')
            return 'C' if vc > vb else 'B'
        if pround == 2:
            vc = self.get_q_s_a(self.q_value2_final, incomplete_card[1:], state, 'C')
            vb = self.get_q_s_a(self.q_value2_final, incomplete_card[1:], state, 'B')
            return 'C' if vc > vb else 'B'

    def dfs(self, pround, ranking, history, invest1, invest2, betting1=2, betting2=4):
        if pround == 1:
            if ranking == 0:
                assert history == ""
                self.round1_states_set.add(history.upper())
                self.player1_states_set.add(history.upper())
                self.dfs(1, ranking + 1, "c", invest1, invest2)
                self.dfs(1, ranking + 1, "b", invest1 + betting1, invest2)
            if ranking == 1:
                self.round1_states_set.add(history.upper())
                self.player2_states_set.add(history.upper())
                self.dfs(pround, ranking + 1, history + "c", invest1, invest2)
                self.dfs(pround, ranking + 1, history + "b", invest1, invest2 + betting1)
            if ranking == 2:
                if history == "bc":
                    if self.verbose:
                        print history, invest1, invest2
                    self.ending[history.upper()] = [invest1, invest2]
                elif history == 'cb':
                    self.round1_states_set.add(history.upper())
                    self.player1_states_set.add(history.upper())
                    self.dfs(pround, ranking + 1, history + "c", invest1, invest2)
                    self.dfs(pround, ranking + 1, history + "b", invest1 + betting1, invest2)
                else:
                    self.player1_states_set.add(history.upper())
                    self.dfs(pround + 1, 1, history + "C", invest1, invest2)
                    self.dfs(pround + 1, 1, history + "B", invest1 + betting2, invest2)
            if ranking == 3:
                if history == "cbc":
                    if self.verbose:
                        print history, invest1, invest2
                    self.ending[history.upper()] = [invest1, invest2]
                else:
                    self.player1_states_set.add(history.upper())
                    self.dfs(pround + 1, 1, history + "C", invest1, invest2)
                    self.dfs(pround + 1, 1, history + "B", invest1 + betting2, invest2)
        if pround == 2:
            if ranking == 1:
                self.player2_states_set.add(history.upper())
                self.dfs(pround, ranking + 1, history + "C", invest1, invest2)
                self.dfs(pround, ranking + 1, history + "B", invest1, invest2 + betting2)
            if ranking == 2:
                if history[-2:] == "CB":
                    self.player1_states_set.add(history.upper())
                    self.dfs(pround, ranking + 1, history + "C", invest1, invest2)
                    self.dfs(pround, ranking + 1, history + "B", invest1 + betting2, invest2)
                else:
                    if self.verbose:
                        print history, invest1, invest2
                    self.ending[history.upper()] = [invest1, invest2]
            if ranking == 3:
                if self.verbose:
                    print history, invest1, invest2
                self.ending[history.upper()] = [invest1, invest2]

    @staticmethod
    def get_card_state_state(mix_state):
        for i in range(len(mix_state)):
            if mix_state[i] == 'C' or mix_state[i] == 'B':
                return mix_state[:i], mix_state[i:]
        return mix_state, ""

    @staticmethod
    def convert_q_s_a2greedy_policy(q_s_a):
        # input is self.q_value1_final or self.q_value2_final
        policy = {}
        for key, item in q_s_a.iteritems():
            p = 1.0 if item['C'] > item['B'] else 0.0
            policy[key] = [p, 1.0 - p]
        return policy

    @staticmethod
    def tournament(seed, games, p1_policy, p2_policy):
        np.random.seed(seed)
        pay1 = pay2 = 0.0
        for i in range(games):
            cards = XFP.possible_cards_list[np.random.randint(len(XFP.possible_cards_list))]
            game_state = ""
            while True:
                if game_state in XFP.ending:
                    v1, v2 = XFP.compute_payoff(cards, game_state)
                    pay1 += v1
                    pay2 += v2
                    break
                if game_state in XFP.player1_states_set:
                    if game_state in XFP.round1_states_set:
                        prob = p1_policy[cards[0] + game_state]
                    else:
                        prob = p1_policy[cards[:2] + game_state]

                    if type(prob) == dict:  # this is a realization
                        prob = [prob['C'][0] / (prob['C'][0] + prob['B'][0]), ]
                    action = 'C' if np.random.rand() < prob[0] else 'B'
                    game_state = game_state + action
                else:
                    assert game_state in XFP.player2_states_set
                    if game_state in XFP.round1_states_set:
                        prob = p2_policy[cards[2] + game_state]
                    else:
                        prob = p2_policy[cards[1:] + game_state]
                    if type(prob) == dict:  # this is a realization
                        prob = [prob['C'][0] / (prob['C'][0] + prob['B'][0]), ]
                    action = 'C' if np.random.rand() < prob[0] else 'B'
                    game_state = game_state + action
        return pay1, pay2

    @staticmethod
    def dfs_realization_forward(realization_func, card_state, state, action, realization, policy1, policy2):
        if state in XFP.ending:
            if XFP.get_q_s_a(realization_func, '', state, card_state) is not None:
                item = XFP.get_q_s_a(realization_func, '', state, card_state)
            else:
                item = [0.0, 0.0]
            item = [item[0] + realization, item[1] + 1.0]
            XFP.update_q_s_a(realization_func, '', state, card_state, item)
            return
        if state in XFP.round1_states_set:
            if state in XFP.player1_states_set:
                card = card_state[0]
            else:
                card = card_state[2]
        else:
            if state in XFP.player1_states_set:
                card = card_state[:2]
            else:
                card = card_state[1:]
        if action != '':
            if XFP.get_q_s_a(realization_func, card, state, action) is not None:
                item = XFP.get_q_s_a(realization_func, card, state, action)
            else:
                item = [0.0, 0.0]
            item = [item[0] + realization, item[1] + 1.0]
            XFP.update_q_s_a(realization_func, card, state, action, item)

        state_next = state + action
        if state_next in XFP.ending:
            XFP.dfs_realization_forward(realization_func, card_state, state_next, '', realization, policy1, policy2)
            return
        elif state_next in XFP.player1_states_set:
            if state_next in XFP.round1_states_set:
                card_next = card_state[0]
            else:
                card_next = card_state[:2]
            prob = policy1[card_next + state_next]
        elif state_next in XFP.player2_states_set:
            if state_next in XFP.round1_states_set:
                card_next = card_state[2]
            else:
                card_next = card_state[1:]
            prob = policy2[card_next + state_next]
        else:
            assert False
        if type(prob) == dict:  #  this is a realization
            prob = [prob['C'][0] / (prob['C'][0] + prob['B'][0]), prob['B'][0] / (prob['C'][0] + prob['B'][0])]
        XFP.dfs_realization_forward(realization_func, card_state, state_next, 'C', prob[0] * realization, policy1, policy2)
        XFP.dfs_realization_forward(realization_func, card_state, state_next, 'B', prob[1] * realization, policy1, policy2)

    @staticmethod
    def compute_realization(policy_p1, policy_p2):
        """ policy_p1 and policy_p2 could be realizations """
        realization_func = {}
        for cards in XFP.possible_cards_list:
            XFP.dfs_realization_forward(realization_func, cards, "", "", 1.0, policy_p1, policy_p2)
        return realization_func

    @staticmethod
    def compute_realization2policy(realization):
        policy1 = {}
        policy2 = {}
        for cards in XFP.possible_cards_list:
            for state in XFP.player1_states_set:
                if state in XFP.round1_states_set:
                    card = cards[0]
                else:
                    card = cards[:2]
                v1 = realization[card + state]['C'][0]
                v2 = realization[card + state]['B'][0]
                policy1[card + state] = [v1 / (v1 + v2), v2 / (v1 + v2)]
            for state in XFP.player2_states_set:
                if state in XFP.round1_states_set:
                    card = cards[2]
                else:
                    card = cards[1:]
                v1 = realization[card + state]['C'][0]
                v2 = realization[card + state]['B'][0]
                policy2[card + state] = [v1 / (v1 + v2), v2 / (v1 + v2)]
        return policy1, policy2

    @staticmethod
    def compute_payoff_given_realization(realization):
        tot = 0.0
        payoff = np.zeros([2], np.float64)
        for key, item in realization.iteritems():
            if key in XFP.ending:
                for card_state, item2 in item.iteritems():
                    payoff += item2[0] * np.asarray(XFP.compute_payoff(card_state, key))
                    tot += item2[0]
        payoff /= tot
        assert tot != 0.0  # if tot == 0 your realization is from mixture which can not be used for payoff computation
        return payoff

    @staticmethod
    def mix_realization(realization_br, realization_old, ratio):
        mix_realization = {}
        for key in realization_br:
            if key in XFP.ending:  # mix_realization does not have ending states thus can not compute payoff
                continue
            assert realization_br[key]['C'][1] == realization_old[key]['C'][1]
            assert realization_br[key]['B'][1] == realization_old[key]['B'][1]
            v1 = [realization_br[key]['C'][0] * ratio + realization_old[key]['C'][0] * (1 - ratio), realization_br[key]['C'][1]]
            v2 = [realization_br[key]['B'][0] * ratio + realization_old[key]['B'][0] * (1 - ratio), realization_br[key]['B'][1]]
            s_v1_v2 = v1[0] + v2[0]
            v1[0] = v1[0] / s_v1_v2
            v2[0] = v2[0] / s_v1_v2
            mix_realization[key] = {
                'C': v1,
                'B': v2}
        return mix_realization


class LeducRLEnv(object):
    history_string2vector = {}

    def __init__(self, verbose=False, card_num=6, seed=None):
        """ state space:
                card vector: 6
                history vector: 2 * 2 * 2 * 2 (players, rounds, max_bets, actions)
        """
        assert XFP.bool_init
        self.state_space = 16+card_num
        self.state_history_space = [2, 2, 2, 2]
        self.state_card_space = [card_num]
        self.action_space = 2
        self.card_num = card_num
        self.bet_num = np.zeros([2, 2], np.int32)
        self.dfs_state_history = np.zeros([2, 2, 2, 2], np.int32)
        if seed is not None:
            np.random.seed(seed)

        self.history_state = np.zeros([2, 2, 2, 2], np.int32)
        self.p0_card_vector = np.zeros([card_num], np.int32)
        self.p1_card_vector = np.zeros([card_num], np.int32)
        self.cards = None
        self.round = 1
        self.current_player = 0
        self.history_string = ""

        self.dfs("")
        if verbose:
            for key in LeducRLEnv.history_string2vector:
                print key, np.reshape(LeducRLEnv.history_string2vector[key], [-1])

    def dfs(self, history):
        if history in XFP.ending:
            return
        if history in LeducRLEnv.history_string2vector:
            assert np.allclose(self.dfs_state_history, LeducRLEnv.history_string2vector[history])
        else:
            LeducRLEnv.history_string2vector[history] = self.dfs_state_history.copy()
        player = 0 if history in XFP.player1_states_set else 1
        pround = 1 if history in XFP.round1_states_set else 2

        self.dfs_state_history[player, pround - 1, self.bet_num[player, pround - 1], 0] = 1
        self.bet_num[player, pround - 1] += 1
        self.dfs(history + 'C')
        self.bet_num[player, pround - 1] -= 1
        self.dfs_state_history[player, pround - 1, self.bet_num[player, pround - 1], 0] = 0
        self.dfs_state_history[player, pround - 1, self.bet_num[player, pround - 1], 1] = 1
        self.bet_num[player, pround - 1] += 1
        self.dfs(history + 'B')
        self.bet_num[player, pround - 1] -= 1
        self.dfs_state_history[player, pround - 1, self.bet_num[player, pround - 1], 1] = 0

    def set_card_vectors(self):
        self.p0_card_vector[int(self.cards[0])] = 1
        self.p1_card_vector[int(self.cards[2])] = 1
        if self.round == 2:
            self.p0_card_vector[int(self.card_num)/2 + int(self.cards[1])] = 1
            self.p1_card_vector[int(self.card_num)/2 + int(self.cards[1])] = 1

    def reset(self, given_cards=None):
        # self.history_state = np.zeros([2, 2, 2, 2], np.int32)
        self.p0_card_vector = np.zeros([self.card_num], np.int32)
        self.p1_card_vector = np.zeros([self.card_num], np.int32)
        self.cards = XFP.possible_cards_list[np.random.randint(len(XFP.possible_cards_list))] if given_cards is None else given_cards
        self.current_player = 0
        self.round = 1
        # self.bet_num = np.zeros([2, 2], np.int32)
        self.set_card_vectors()
        self.history_string = ''

        observation = {'card': self.p0_card_vector,
                       'state': self.history_state,
                       'turn': 0,
                       'card_str': self.cards[0],
                       'history_str': self.history_string}
        return observation

    def act(self, action):
        if action == 0:
            self.history_string += 'C'
            # self.history_state[self.current_player, self.round - 1, self.bet_num[self.current_player, self.round - 1], 0] = 1
        else:
            self.history_string += 'B'
            # self.history_state[self.current_player, self.round - 1, self.bet_num[self.current_player, self.round - 1], 1] = 1
        # self.bet_num[self.current_player, self.round - 1] += 1

        if self.history_string in XFP.ending:
            p1_payoff, p2_payoff = XFP.compute_payoff(self.cards, self.history_string)
            return {'turn': -1, 'payoff': [p1_payoff, p2_payoff], 'history_str': self.history_string}
        self.round = 1 if self.history_string in XFP.round1_states_set else 2
        self.set_card_vectors()
        if self.history_string in XFP.player1_states_set:
            self.current_player = 0
            card_str = self.cards[0] if self.round == 1 else self.cards[:2]
            return {'card': self.p0_card_vector,
                    'state': LeducRLEnv.history_string2vector[self.history_string],
                    'turn': 0,
                    'card_str': card_str,
                    'history_str': self.history_string}
        else:
            self.current_player = 1
            card_str = self.cards[2] if self.round == 1 else self.cards[1:]
            return {'card': self.p1_card_vector,
                    'state': LeducRLEnv.history_string2vector[self.history_string],
                    'turn': 1,
                    'card_str': card_str,
                    'history_str': self.history_string}


if __name__ == "__main__":
    env_xfp = XFP(True)
    env = LeducRLEnv(True, seed=100)

    # for i in range(1000):
    #     ob = env.reset()
    #     while ob['turn'] != -1:
    #         ob = env.act(np.random.randint(0, 2))
    #     print ob

