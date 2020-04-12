import math as m
import numpy as np
import game


class MonteCarloTreeSearch:

    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct    # used to tweak exploration rate
        self.visit_count = {}   # [N(s, a)]
        self.value = {}         # [W(s, a)], total value
        self.value_avg = {}     # [Q(s, a)], average value
        self.probs = {}         # [P(s, a)], prior probabilities

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def find_leaf(self, state_int: int, player: int):
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)
            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            if cur_state == state_int:
                noises = np.random.dirichlet([0.03] * game.GAME_COLS)
                probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)]
            score = [value + self.c_puct * prob * total_sqrt / (1 + count)
                     for value, prob, count in zip(values_avg, probs, counts)]

            invalid_actions = set(range(game.GAME_COLS)) - set(game.possible_moves(cur_state))
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            actions.append(action)


if __name__ == '__main__':
    pass
