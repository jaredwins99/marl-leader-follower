# source activate py3.8
import pygame
import os,sys,subprocess, logging
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from joblib import dump,load

# Custom
from s01_agents import *
from s02_environment import *


# Initialize the argument parser
parser = argparse.ArgumentParser(description="A script that trains multiple agent Q-learning.")

# Define arguments as a dictionary
arguments = {
    "iter": {"type": int, "default": 10000, "help": "The total number of iterations."},
    "epsilon": {"type": float, "help": "Exploration amount."},
    "a1_target_reward": {"type": float, "help": "Reward agent 1 receives at target."},
    "a2_target_reward": {"type": float, "help": "Reward agent 2 receives at target."},
    "together_reward": {"type": float, "default": 0, "help": "Reward for staying together."},
    "travel_reward": {"type": float, "help": "Reward for traveling one step (negative)."},
    "prior_q": {"type": str, "default": "", "help": "File name of previous Q function (saved as *.pkl)."},
    "save_dir": {"type": str, "help": "Directory to save results."},
    "epsilon_eval": {"type": float, "help": "Exploration amount for evaluation."},
    "train": {"action": "store_true", "help": "Train the agent."},
    "plot": {"action": "store_true", "help": "Plot the Q values."},
    "eval": {"type": int, "default": 0, "help": "Number of trials to evaluate."},
    "plot_grid": {"action": "store_true", "help": "Whether to plot in a grid or not."},
    "agent": {"type": str, "default": "JointAgent", "help": "Which agent to use."},
    "predictor": {"type": str, "default": "", "help": "Which predictor to use in agent prediction."},
    "grid_size": {"type": int, "default": 5, "help": "Size of the grid."}
}

# Add arguments to the parser
for arg, params in arguments.items():
    parser.add_argument(f'--{arg}', **params)

# Parse arguments
args = parser.parse_args()

# Set up the environment
env_set = {
    'epsilon': args.epsilon,
    'a1_target_reward': args.a1_target_reward,
    'a2_target_reward': args.a2_target_reward,
    'together_reward': args.together_reward,
    'travel_reward': args.travel_reward,
    'NUM_FRAMES': args.iter,
    'save_dir': f'experiments/{args.save_dir}/',
    'plot_grid': args.plot_grid,
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0),
    'BLUE': (0, 0, 255),
    'GREEN': (0, 255, 0),
    'BLACK': (0, 0, 0),
    'FPS': 120,
    'RED1': (179, 0, 0),
    'RED2': (127, 0, 0),
    'WIDTH': 500,
    'HEIGHT': 500,
    'grid_size': args.grid_size,
}

env_set['reward_position1'] = (0, env_set['grid_size'] // 2)
env_set['reward_position2'] = (env_set['grid_size'] - 1, env_set['grid_size'] // 2)
env_set['cell_width'] = env_set['WIDTH'] / env_set['grid_size']
env_set['cell_height'] = env_set['HEIGHT'] / env_set['grid_size']

if not os.path.exists(env_set['save_dir']):
    os.mkdir(env_set['save_dir'])

# Create a log file in the save directory
log_file = os.path.join(env_set['save_dir'], 'log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO)

agent = eval(args.agent)(env_set['grid_size'], env_set['grid_size'], draw_radius=20, alpha=0.1, gamma=0.9, epsilon=env_set['epsilon'])
if args.predictor:
    agent.set_predictor(args.predictor)
agent.reset_to_center()

# Add priors
if args.prior_q:
    with open(args.prior_q, 'rb') as f:
        qval = pickle.load(f)
    agent.q_values1 = qval['q_val1']
    agent.q_values2 = qval['q_val2']

if args.train:
    agent, rewards_activated, rewards_collected, cumulative_rewards = train_qval(agent, env_set)
    rewards = {'rewards_activated': rewards_activated, 'rewards_collected': rewards_collected, 'cumulative_rewards': cumulative_rewards}
    with open(env_set['save_dir']+'rewards.pkl', 'wb') as f:
        pickle.dump(rewards, f)
    qvals = {'q_val1': agent.q_values1, 'q_val2': agent.q_values2}
    with open(env_set['save_dir']+'q_val.pkl', 'wb') as f:
        pickle.dump(qvals, f)
    if hasattr(agent,'predictor') and agent.predictor:
        logging.info(f'Predictor correct probability: {agent.right_prediction / env_set["NUM_FRAMES"]}')
        predictor_stats = {'accuracy': agent.accuracy,'rp_length': agent.rp_length}
        with open(env_set['save_dir']+'predictor_stats.pkl', 'wb') as f:
            pickle.dump(predictor_stats, f)
        dump(agent.predictor, env_set['save_dir']+'predictor.joblib')

if args.eval:
    agent.epsilon = args.epsilon_eval
    if hasattr(agent,'predictor') and agent.predictor:
        agent.predictor = load(env_set['save_dir']+'predictor.joblib')
        agent.predictor_built = True
    seq = []
    success_series = {'a1' : [], 'a2' : []}
    rewards_activated = {'a1' : [], 'a2' : []}
    rewards_collected = {'a1' : [], 'a2' : []}
    zones1_counted = {'a1' : [],
                     'a2' : []}
    zones2_counted = {'a1' : [],
                      'a2' : []}
    zones3_counted = {'a1' : [],
                      'a2' : []}
    zones4_counted = {'a1' : [],
                      'a2' : []}
    times_to_activation = {'a1' : [], 'a2' : []}
    times_to_reward = {'a1' : [], 'a2' : []}
    deviations_from_reward = {'a1' : [], 'a2' : []}
    deviations_from_activation = {'a1' : [], 'a2' : []}
    for i in range(args.eval):
        agent.reset()
        (trajectory, 
         success_series,
         rewards_activated, 
         rewards_collected, 
         zones1_counted,
         zones2_counted,
         zones3_counted,
         zones4_counted, 
         times_to_activation, 
         times_to_reward,
         deviations_from_reward, 
         deviations_from_activation) = eval_qval(agent, 
                                                 env_set, 
                                                 success_series,
                                                 rewards_activated, 
                                                 rewards_collected, 
                                                 zones1_counted, 
                                                 zones2_counted,
                                                 zones3_counted,
                                                 zones4_counted,
                                                 times_to_activation,
                                                 times_to_reward, 
                                                 deviations_from_reward, 
                                                 deviations_from_activation)
        seq.append(trajectory)
    rewards = {'success': [success_series],
               'rewards_activated': [rewards_activated], 
               'rewards_collected': [rewards_collected],
               'zones1_counted': [zones1_counted],
               'zones2_counted': [zones2_counted],
               'zones3_counted': [zones3_counted],
               'zones4_counted': [zones4_counted],
               'times_to_activation': [times_to_activation],
               'times_to_reward': [times_to_reward],
               'deviations_from_reward': [deviations_from_reward],
               'deviations_from_activation': [deviations_from_activation]}
    with open(env_set['save_dir'] + 'eval.pkl', 'wb') as f:
        pickle.dump(seq, f)
    with open(env_set['save_dir'] + 'rewards_eval.pkl', 'wb') as f:
        pickle.dump(rewards, f)

if args.plot:
    plot_qval(agent, env_set)
    if hasattr(agent,'predictor') and agent.predictor:
        plot_predictor(agent, env_set)

# # Log the input arguments
# logging.info('Input Arguments:')
# for arg in vars(args):
#     logging.info(f'{arg}: {getattr(args, arg)}')
