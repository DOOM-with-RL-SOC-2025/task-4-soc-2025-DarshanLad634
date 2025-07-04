from utils import pos_to_xy, xy_to_pos, in_bounds, get_line_between
from collections import defaultdict

class FootballEnv:
    def __init__(self, p, q, opponent_policy):
        self.p = p  # probability parameter for movement failure
        self.q = q  # probability parameter for pass/shoot success
        self.policy = opponent_policy  # fixed opponent policy (dict)
        self.actions = list(range(10))  # 0-3 for B1 move, 4-7 for B2 move, 8 = pass, 9 = shoot
        self.states = self.get_all_states()

    def get_all_states(self):
        """
        Enumerates all possible legal game states.
        """
        states = []
        for b1 in range(1, 17):
            for b2 in range(1, 17):
                if b2 == b1: continue
                for r in range(1, 17):
                    if r in (b1, b2): continue
                    for ball in [1, 2]:
                        states.append((b1, b2, r, ball))
        states.append((-1, -1, -1, -1))  # terminal state
        return states

    def is_terminal(self, state):
        return state == (-1, -1, -1, -1)

    def move(self, pos, direction):
        """
        Attempts to move a player in a given direction from a position.
        Returns -1 if out of bounds.
        """
        x, y = pos_to_xy(pos)
        if direction == 'L': x -= 1
        elif direction == 'R': x += 1
        elif direction == 'U': y -= 1
        elif direction == 'D': y += 1
        return xy_to_pos(x, y) if in_bounds(x, y) else -1

    def get_opponent_transitions(self, state):
        """
        Returns possible next positions for opponent R based on its policy and action probabilities.
        """
        probs = self.policy.get(state, [0.25] * 4)
        dirs = ['L', 'R', 'U', 'D']
        transitions = []
        _, _, r, _ = state
        for i, prob in enumerate(probs):
            new_r = self.move(r, dirs[i])
            transitions.append((prob, new_r if new_r != -1 else r))
        return transitions

    def get_transitions(self, state, action):
        """
        Computes next-state transitions given a state and an action.
        Returns a list of (probability, next_state, reward) tuples.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0)]

        b1, b2, r, ball = state
        result = []
        ball_pos = b1 if ball == 1 else b2
        dirs = ['L', 'R', 'U', 'D']

        def end_state(): return (-1, -1, -1, -1)

        for prob_r, new_r in self.get_opponent_transitions(state):

            if action in range(8):  # movement
                is_b1 = action < 4
                actor = b1 if is_b1 else b2
                dir_idx = action % 4
                move_dir = dirs[dir_idx]
                new_pos = self.move(actor, move_dir)

                if new_pos == -1:
                    result.append((prob_r, end_state(), 0))
                    continue

                if (is_b1 and ball == 1) or (not is_b1 and ball == 2):  # with ball
                    fail_prob = 2 * self.p
                    success_prob = 1 - fail_prob

                    if new_pos == new_r:  # tackle case A
                        result.append((prob_r * success_prob * 0.5, (new_pos if is_b1 else b1,
                                                                    new_pos if not is_b1 else b2,
                                                                    new_r, ball), 0))
                        result.append((prob_r * (1 - success_prob * 0.5), end_state(), 0))
                    elif actor == new_r and new_pos == r:  # tackle case B
                        result.append((prob_r * success_prob * 0.5, (new_pos if is_b1 else b1,
                                                                    new_pos if not is_b1 else b2,
                                                                    new_r, ball), 0))
                        result.append((prob_r * (1 - success_prob * 0.5), end_state(), 0))
                    else:
                        result.append((prob_r * success_prob, (new_pos if is_b1 else b1,
                                                              new_pos if not is_b1 else b2,
                                                              new_r, ball), 0))
                        result.append((prob_r * fail_prob, end_state(), 0))

                else:  # moving without ball
                    fail_prob = self.p
                    success_prob = 1 - fail_prob
                    result.append((prob_r * success_prob, (new_pos if is_b1 else b1,
                                                          new_pos if not is_b1 else b2,
                                                          new_r, ball), 0))
                    result.append((prob_r * fail_prob, end_state(), 0))

            elif action == 8:  # pass
                passer = b1 if ball == 1 else b2
                receiver = b2 if ball == 1 else b1
                dist = max(abs(pos_to_xy(passer)[0] - pos_to_xy(receiver)[0]),
                           abs(pos_to_xy(passer)[1] - pos_to_xy(receiver)[1]))
                pass_prob = max(self.q - 0.1 * dist, 0)
                line = get_line_between(passer, receiver)
                if new_r in line or new_r == passer or new_r == receiver:
                    pass_prob *= 0.5
                result.append((prob_r * pass_prob, (b1, b2, new_r, 2 if ball == 1 else 1), 0))
                result.append((prob_r * (1 - pass_prob), end_state(), 0))

            elif action == 9:  # shoot
                shooter = b1 if ball == 1 else b2
                x, _ = pos_to_xy(shooter)
                shot_prob = max(self.q - 0.2 * (3 - x), 0)
                if new_r in [8, 12]:  # opponent blocking goal
                    shot_prob *= 0.5
                result.append((prob_r * shot_prob, end_state(), 1))  # goal
                result.append((prob_r * (1 - shot_prob), end_state(), 0))  # miss

        return result
