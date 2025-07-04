import argparse
import json
import os
import matplotlib.pyplot as plt
from environment import FootballEnv
from agent import Agent
from utils import pretty_print

# Load opponent policy file (JSON) into a usable dict
def load_policy(path):
    with open(path, 'r') as f:
        return json.load(f)

# Get the expected value from a specific start state
def run_episode(env, agent, start_state):
    return agent.V.get(start_state, 0.0)

# Plot and save a graph
def plot_graph(xs, ys, xlabel, ylabel, title, filename):
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, default=None, help='Parameter p ∈ [0, 0.5]')
    parser.add_argument('--q', type=float, default=None, help='Parameter q ∈ [0.6, 1]')
    parser.add_argument('--policy', type=str, required=True,
                        choices=['greedy', 'park', 'random'], help='Opponent strategy')
    parser.add_argument('--print', action='store_true', help='Pretty print the start state')
    parser.add_argument('--graph', action='store_true', help='Plot graphs instead of single eval')
    args = parser.parse_args()

    # Load opponent policy file
    policy_file = f"data/{args.policy}_policy.json"
    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"Policy file not found: {policy_file}")

    policy_data = load_policy(policy_file)
    opponent_policy = {tuple(map(int, k.strip('[]').split(','))): v for k, v in policy_data.items()}

    start_state = (5, 9, 8, 1)  # Default start state

    if args.graph:
        # Generate Graph 1: varying p, fixed q = 0.7
        ps = [round(i * 0.1, 1) for i in range(6)]
        results1 = []
        for p in ps:
            env = FootballEnv(p, 0.7, opponent_policy)
            agent = Agent(env)
            agent.value_iteration()
            win_prob = run_episode(env, agent, start_state)
            results1.append(win_prob)
        plot_graph(ps, results1, 'p', 'Probability of Goal', 'Goal Prob vs p (q=0.7)', 'graph_p_vs_goal.png')

        # Generate Graph 2: varying q, fixed p = 0.3
        qs = [round(0.6 + 0.1 * i, 1) for i in range(5)]
        results2 = []
        for q in qs:
            env = FootballEnv(0.3, q, opponent_policy)
            agent = Agent(env)
            agent.value_iteration()
            win_prob = run_episode(env, agent, start_state)
            results2.append(win_prob)
        plot_graph(qs, results2, 'q', 'Probability of Goal', 'Goal Prob vs q (p=0.3)', 'graph_q_vs_goal.png')

        print("Graphs saved: graph_p_vs_goal.png and graph_q_vs_goal.png")
        return

    # If --graph is not given, evaluate a single configuration
    if args.p is None or args.q is None:
        raise ValueError("Must specify both --p and --q if not using --graph")

    # Run single evaluation
    env = FootballEnv(args.p, args.q, opponent_policy)
    agent = Agent(env)
    agent.value_iteration()
    win_prob = run_episode(env, agent, start_state)
    print(f"Expected goal probability from state {start_state}: {win_prob:.4f}")

    # Optional: pretty print state + chosen action
    if args.print:
        print("\nInitial State:")
        pretty_print(start_state)
        action = agent.policy.get(start_state, None)
        print(f"\nChosen Action: {action}")

if __name__ == '__main__':
    main()
