from collections import defaultdict

class Agent:
    def __init__(self, env, gamma=1.0, eps=1e-4):
        self.env = env                  # Reference to environment
        self.gamma = gamma              # Discount factor
        self.eps = eps                  # Convergence threshold
        self.V = defaultdict(float)     # Value function (maps state → value)
        self.policy = {}                # Learned policy (maps state → best action)

    def value_iteration(self):
        """
        Core algorithm for computing the optimal policy using Value Iteration.
        Iterates over all states and updates values and policy until convergence.
        """
        iteration = 0
        while True:
            delta = 0  # Tracks the maximum value change in this iteration
            for state in self.env.states:
                if self.env.is_terminal(state):
                    self.V[state] = 0.0
                    continue

                max_val = float('-inf')  # Best value across all actions
                best_act = None

                for a in self.env.actions:
                    total = 0  # Expected value for taking action `a` at `state`
                    for prob, next_state, reward in self.env.get_transitions(state, a):
                        total += prob * (reward + self.gamma * self.V[next_state])

                    # Update if this action has better value
                    if total > max_val:
                        max_val = total
                        best_act = a

                # Track maximum change in values
                delta = max(delta, abs(self.V[state] - max_val))
                self.V[state] = max_val
                self.policy[state] = best_act

            iteration += 1
            if delta < self.eps:
                break  # Converged
        return self.V, self.policy
