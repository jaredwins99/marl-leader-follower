import pygame
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import pickle
from random import sample

# Custom
from s01_agents import *


def train_qval(agent, env_set):
    """
    Train the agent using the specified environment settings. (remove the plotting option)

    Args:
        agent (Agent): The agent to be trained.
        env_set (dict): The environment settings.

    Returns:
        tuple: A tuple containing the trained agent, the count of rewards obtained by both agents together, and the count of rewards obtained by both agents separately.
    """
    
    for key, value in env_set.items():
        globals()[key] = value
    
    reward_place_to_coord = {
        () : (),
        ('up') : ((grid_size//2,grid_size-1)),
        ('right') : ((grid_size-1,grid_size//2)),
        ('down') : ((grid_size//2,0)),
        ('left') : ((0,grid_size//2)),
        ('up','right') : ((grid_size//2,grid_size-1),(grid_size-1,grid_size//2)),
        ('right','down') : ((grid_size-1,grid_size//2),(grid_size//2,0)),
        ('down','left') : ((grid_size//2,0),(0,grid_size//2)),
        ('up','left') : ((0,grid_size//2),(grid_size//2,grid_size-1)),
        ('up','down') : ((grid_size//2,grid_size-1),(grid_size//2,0)),
        ('right','left') : ((grid_size-1,grid_size//2),(0,grid_size//2)),
        ('up','right','down') : ((grid_size//2,grid_size-1),(grid_size-1,grid_size//2),(grid_size//2,0)),
        ('right','down','left') : ((grid_size-1,grid_size//2),(grid_size//2,0),(0,grid_size//2)),
        ('up','down','left') : ((grid_size//2,0),(0,grid_size//2),(grid_size//2,grid_size-1)),
        ('up','right','left') : ((0,grid_size//2),(grid_size//2,grid_size-1),(grid_size-1,grid_size//2)),
        ('up','right','down','left') : ((grid_size//2,grid_size-1),(grid_size-1,grid_size//2),(grid_size//2,0),(0,grid_size//2))
    }  
    
    def train_regime(agent, regime, iteration_multiplier=1):
        
        rewards_activated_series = {'a1' : [], 'a2' : []}
        rewards_collected_series = {'a1' : [], 'a2' : []}
        cumulative_reward_series = {'a1' : [], 'a2' : []}
        running = True
        frame_count = 0
        consecutive_noreward = 0
        rp  = []
        rewards_collected_count = 0
        a1_rewards_activated_count = 0
        a2_rewards_activated_count = 0
        true_reward_locations = ('right','left')
        while running and frame_count < NUM_FRAMES * iteration_multiplier:
            
            agent.epsilon = (0.4-env_set['epsilon'])/(NUM_FRAMES * iteration_multiplier) * frame_count + env_set['epsilon']
        
            reward1 = reward2 = 0
            rewards_collected_count = 0
            a1_rewards_activated_count = a2_rewards_activated_count = 0
            
            # reset if no reward for a long time
            if consecutive_noreward > 50:
                agent.reset()
                consecutive_noreward = 0
                if regime == 0:
                    true_reward_locations = ('right','left')
                if regime == 1:
                    true_reward_locations = ('right','left')
                if regime == 2:
                    true_reward_locations = ('up','down')
                if regime == 3:
                    true_reward_locations = sample([('up','right'),('right','down'),('down','left'),('up','left'),('right','left'),('up','down')], 1)[0]
            
            # Take set difference to determine the other
            non_reward_locations = tuple(set(['up','right','down','left']) - set(true_reward_locations))
            
            # Sort acorrding to scheme up right down left (clockwise)
            non_reward_locations = tuple(sorted(non_reward_locations, key=lambda x: ['up','right','down','left'].index(x)))
            
            # Agent chooses action
            action1, action2 = agent.choose_action()
            if hasattr(agent, 'explore') and not agent.explore:
                rp.append([agent.get_state(0), agent.get_state(1), action1, action2])
            agent.move(0, action1)
            agent.move(1, action2)
            agent.update_occupancy()
            state1 = agent.get_state(0)
            pos1, _ = state1
            state2 = agent.get_state(1)
            pos2, _ = state2
            
            center_pos = (env_set['grid_size'] // 2, env_set['grid_size']  // 2)
            a1_at_center = pos1 == center_pos
            a2_at_center = pos2 == center_pos
            if a1_at_center and not agent.rewards_active: 
                if regime == 0:
                    agent.rewards_active = true_reward_locations
                    reward1 += target_reward1
                    reward2 += target_reward2
                    consecutive_noreward = 999 # dummy number used to reset
                if regime == 1 or regime == 2 or regime == 3:
                    agent.rewards_active = true_reward_locations
                    a1_rewards_activated_count = 1
                
            if a2_at_center and not agent.rewards_active:
                if regime == 0:
                    agent.rewards_active = true_reward_locations
                    reward1 += target_reward1
                    reward2 += target_reward2
                    consecutive_noreward = 999 # dummy number used to reset
                if regime == 1 or regime == 2 or regime == 3:
                    agent.rewards_active = true_reward_locations
                    a2_rewards_activated_count = 1

            
            #print(reward_place_to_coord[agent.rewards_active])
            #print(reward_place_to_coord[agent.rewards_active])
            
            # Calculate reward based on the next state
            if ((agent.rewards_active and 
                 pos1 == reward_place_to_coord[true_reward_locations][0] and 
                 pos2 == reward_place_to_coord[true_reward_locations][0]) or 
                (agent.rewards_active and 
                 pos1 == reward_place_to_coord[true_reward_locations][1] and 
                 pos2 == reward_place_to_coord[true_reward_locations][1])):
                if regime == 1 or regime == 2 or regime == 3:
                    reward1 += target_reward1
                    reward2 += target_reward2
                    reward_collected_count = 1
                    consecutive_noreward = 999 # dummy number used to reset
            
            if ((agent.rewards_active and 
                 pos1 == reward_place_to_coord[non_reward_locations][0] and 
                 pos2 == reward_place_to_coord[non_reward_locations][0]) or 
                (agent.rewards_active and 
                 pos1 == reward_place_to_coord[non_reward_locations][1] and 
                 pos2 == reward_place_to_coord[non_reward_locations][1])):
                consecutive_noreward = 999 # dummy number used to reset
        
            if pos1 == pos2:
                reward1 += together_reward
                reward2 += together_reward
                
            reward1 += travel_reward # energy loss
            reward2 += travel_reward
            consecutive_noreward += 1
                
            # Update Q-value with Bellman Equation
            if agent.previous_state is not None:
                agent.update_q_value(agent.get_state(0), agent.get_state(1), reward1, reward2)
            agent.previous_state = (agent.get_state(0), agent.get_state(1))
            agent.previous_action = (action1, action2)
            agent.previous_q_values = agent.q_values2.copy()
            frame_count += 1
            
            # clear replay buffer
            if len(rp) > 5000: 
                if agent.predictor is not None:
                    rp_state1 = [ele[0] for ele in rp]
                    rp_state2 = [ele[1] for ele in rp]
                    rp_action2 = [ele[3] for ele in rp]
                    agent.build_predictor(np.concatenate((rp_state1,rp_state2),axis=1), np.ravel(rp_action2))
                rp = []
                
            # Print progress
            if frame_count % 10000 == 0:
                print(f"Frame {frame_count} completed")
            
            rewards_activated_series['a1'].append(a1_rewards_activated_count)
            rewards_activated_series['a2'].append(a2_rewards_activated_count)
            rewards_collected_series['a1'].append(rewards_collected_count)
            rewards_collected_series['a2'].append(rewards_collected_count)
            cumulative_reward_series['a1'].append(reward1)
            cumulative_reward_series['a2'].append(reward2)
        
        return agent, rewards_activated_series, rewards_collected_series, cumulative_reward_series
    
    training_regimes = [0, 1, 2, 3]
    multipliers = [1, 1, 1, 1]
    regime_details = {regime : multiplier for regime, multiplier in zip(training_regimes, multipliers)}
    rewards_activated = []
    rewards_collected = []
    cumulative_rewards = []
    for regime, multiplier in regime_details.items():
        agent, reward_activated_series, reward_time_series, cumulative_reward_series = train_regime(agent, regime, multiplier)
        rewards_activated.append(reward_activated_series)
        rewards_collected.append(reward_time_series)
        cumulative_rewards.append(cumulative_reward_series)
    
    return agent, rewards_activated, rewards_collected, cumulative_rewards


def zone_rec(reward, radius):
    """
    Captures a rectangular region around the reward.
    Args:
        reward: A tuple (r, c) representing the reward position.
        radius: The distance from the reward to the rectangle's boundary.
    Returns:
        A set of tuples representing the cells in the rectangular zone.
    """
    r, c = reward
    cells = set()
    for dr in range(-radius, radius + 1):  # Row offset
        for dc in range(-radius, radius + 1):  # Column offset
            nr, nc = r + dr, c + dc  # New row and column
            if 0 <= nr < 11 and 0 <= nc < 11:  # Stay within grid bounds
                cells.add((nr, nc))
    return cells


def zone_dist(reward, radius):
    """
    Captures a region around the reward based on a given Manhattan distance.
    Args:
        reward: A tuple (r, c) representing the reward position.
        radius: The distance from the reward to the region's boundary.
    Returns:
        A set of tuples representing the cells in the distance-based zone.
    """
    r, c = reward
    cells = set()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) <= radius:  # Manhattan distance ≤2
                nr, nc = r + dr, c + dc
                if 0 <= nr < 11 and 0 <= nc < 11:
                    cells.add((nr, nc))
    return cells


def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def eval_qval(agent, 
              env_set, 
              success_series,
              rewards_activated_series, 
              rewards_collected_series, 
              zone1_count_series,
              zone2_count_series,
              zone3_count_series,
              zone4_count_series, 
              time_to_activation_series, 
              time_to_reward_series,
              deviation_from_activation_series,
              deviation_from_reward_series
              ):
    
    for key, value in env_set.items():
        globals()[key] = value
        
    reward_place_to_coord = {
        () : (),
        ('up') : ((grid_size//2,grid_size-1)),
        ('right') : ((grid_size-1,grid_size//2)),
        ('down') : ((grid_size//2,0)),
        ('left') : ((0,grid_size//2)),
        ('up','right') : ((grid_size//2,grid_size-1),(grid_size-1,grid_size//2)),
        ('right','down') : ((grid_size-1,grid_size//2),(grid_size//2,0)),
        ('down','left') : ((grid_size//2,0),(0,grid_size//2)),
        ('up','left') : ((0,grid_size//2),(grid_size//2,grid_size-1)),
        ('up','down') : ((grid_size//2,grid_size-1),(grid_size//2,0)),
        ('right','left') : ((grid_size-1,grid_size//2),(0,grid_size//2)),
        ('up','right','down') : ((grid_size//2,grid_size-1),(grid_size-1,grid_size//2),(grid_size//2,0)),
        ('right','down','left') : ((grid_size-1,grid_size//2),(grid_size//2,0),(0,grid_size//2)),
        ('up','down','left') : ((grid_size//2,0),(0,grid_size//2),(grid_size//2,grid_size-1)),
        ('up','right','left') : ((0,grid_size//2),(grid_size//2,grid_size-1),(grid_size-1,grid_size//2)),
        ('up','right','down','left') : ((grid_size//2,grid_size-1),(grid_size-1,grid_size//2),(grid_size//2,0),(0,grid_size//2))
    }  
            
    running = True
    frame_count = 0
    consecutive_noreward = 0
    seq = []
    true_reward_locations = sample([('up','right'),('right','down'),('down','left'),('up','left'),('right','left'),('up','down')], 1)[0]
    starting_pos1, _ = agent.get_state(0)
    starting_pos2, _ = agent.get_state(1)
    a1_num_moves = 0
    a2_num_moves = 0
    a1_num_moves_activation = 0
    a2_num_moves_activation = 0
    a1_zone1_counts = []
    a2_zone1_counts = []
    a1_zone2_counts = []
    a2_zone2_counts = []
    a1_zone3_counts = []
    a2_zone3_counts = []
    a1_zone4_counts = []
    a2_zone4_counts = []
    success = 0
    rewards_collected_count = 0
    a1_d1 = manhattan_distance(starting_pos1, reward_place_to_coord[true_reward_locations][0])
    a1_d2 = manhattan_distance(starting_pos1, reward_place_to_coord[true_reward_locations][1])
    a2_d1 = manhattan_distance(starting_pos2, reward_place_to_coord[true_reward_locations][0])
    a2_d2 = manhattan_distance(starting_pos2, reward_place_to_coord[true_reward_locations][1])
    a1_optimal_to_reward = min(a1_d1, a1_d2)
    a2_optimal_to_reward = min(a2_d1, a2_d2)
    a1_optimal_to_activation = manhattan_distance(starting_pos1, (env_set['grid_size']//2,env_set['grid_size']//2))
    a2_optimal_to_activation = manhattan_distance(starting_pos2, (env_set['grid_size']//2,env_set['grid_size']//2))
    a1_rewards_activated_count = 0
    a2_rewards_activated_count = 0
    a1_time_to_activation = 0
    a2_time_to_activation = 0
    a1_time_to_reward = 0
    a2_time_to_reward = 0
    while running and frame_count < NUM_FRAMES:
        a1_zone1_count = 0
        a2_zone1_count = 0
        a1_zone2_count = 0
        a2_zone2_count = 0
        a1_zone3_count = 0
        a2_zone3_count = 0
        a1_zone4_count = 0
        a2_zone4_count = 0

        

        # Agent chooses action
        action1, action2 = agent.choose_action()
        state1 = agent.get_state(0)
        state2 = agent.get_state(1)
        seq.append([state1, state2, action1, action2, reward_place_to_coord[agent.rewards_active]])
        agent.move(0, action1)
        agent.move(1, action2)
        state1 = agent.get_state(0)
        pos1, rewards_active = state1
        state2 = agent.get_state(1)
        pos2, _ = state2
        
        if action1 != "stay":
            a1_num_moves += 1
        if action2 != "stay":
            a2_num_moves += 1
        
        center_pos = (env_set['grid_size'] // 2, env_set['grid_size']  // 2)
        a1_at_center = pos1 == center_pos
        a2_at_center = pos2 == center_pos
        if a1_at_center and not agent.rewards_active: 
            agent.rewards_active = true_reward_locations
            a1_rewards_activated_count = 1
            a1_num_moves_activation = a1_num_moves
            a1_time_to_activation = frame_count
        if a2_at_center and not agent.rewards_active:
            agent.rewards_active = true_reward_locations
            a2_rewards_activated_count = 1
            a2_num_moves_activation = a2_num_moves
            a2_time_to_activation = frame_count
        
        zone1 = set()
        for reward_default in reward_place_to_coord[('up','right','down','left')]:
            zone1.update(zone_dist(reward_default, 1))
        
        zone2 = set()
        for reward_default in reward_place_to_coord[('up','right','down','left')]:
            zone2.update(zone_rec(reward_default, 1))
        
        zone3 = set()
        for reward_default in reward_place_to_coord[('up','right','down','left')]:
            zone3.update(zone_dist(reward_default, 2))
            
        zone4 = set()
        for reward_default in reward_place_to_coord[('up','right','down','left')]:
            zone4.update(zone_rec(reward_default, 2))

        if pos1 in zone1:
            a1_zone1_count = 1
        if pos2 in zone1:
            a2_zone1_count = 1
        if pos1 in zone2:
            a1_zone2_count = 1
        if pos2 in zone2:
            a2_zone2_count = 1
        if pos1 in zone3:
            a1_zone3_count = 1
        if pos2 in zone3:
            a2_zone3_count = 1
        if pos1 in zone4:
            a1_zone4_count = 1
        if pos2 in zone4:
            a2_zone4_count = 1
        a1_zone1_counts.append(a1_zone1_count)
        a2_zone1_counts.append(a2_zone1_count)
        a1_zone2_counts.append(a1_zone2_count)
        a2_zone2_counts.append(a2_zone2_count)
        a1_zone3_counts.append(a1_zone3_count)
        a2_zone3_counts.append(a2_zone3_count)
        a1_zone4_counts.append(a1_zone4_count)
        a2_zone4_counts.append(a2_zone4_count)
            
        
        # Calculate reward based on the next state
        if (agent.rewards_active and pos1 == reward_place_to_coord[true_reward_locations][0] and pos2 == reward_place_to_coord[true_reward_locations][0]):
            seq.append([state1, state2, None, None, reward_place_to_coord[agent.rewards_active]])
            agent.reset()
            running = False
            a1_time_to_reward = frame_count
            a2_time_to_reward = frame_count
            rewards_collected_count = 1
            success = 1
        elif (agent.rewards_active and pos1 == reward_place_to_coord[true_reward_locations][1] and pos2 == reward_place_to_coord[true_reward_locations][1]):
            seq.append([state1, state2, None, None, reward_place_to_coord[agent.rewards_active]])
            agent.reset()
            running = False
            a1_time_to_reward = frame_count
            a2_time_to_reward = frame_count
            rewards_collected_count = 1
            success = 1
        else:
            consecutive_noreward += 1
        if consecutive_noreward > 50:
            agent.reset()
            consecutive_noreward = 0
            running = False
            true_reward_locations = sample([('up','right'),('right','down'),('down','left'),('up','left'),('right','left'),('up','down')], 1)[0]
        frame_count += 1
        
    a1_deviation_from_reward = a1_num_moves - a1_optimal_to_reward
    a2_deviation_from_reward = a2_num_moves - a2_optimal_to_reward
    a1_deviation_from_activation = a1_num_moves_activation  - a1_optimal_to_activation
    a2_deviation_from_activation = a2_num_moves_activation - a2_optimal_to_activation
    success_series['a1'].append(success)
    success_series['a2'].append(success)
    rewards_activated_series['a1'].append(a1_rewards_activated_count)
    rewards_activated_series['a2'].append(a2_rewards_activated_count)
    rewards_collected_series['a1'].append(rewards_collected_count)
    rewards_collected_series['a2'].append(rewards_collected_count)
    zone1_count_series['a1'].append(np.mean(a1_zone1_counts))
    zone1_count_series['a2'].append(np.mean(a2_zone1_counts))
    zone2_count_series['a1'].append(np.mean(a1_zone2_counts))
    zone2_count_series['a2'].append(np.mean(a2_zone2_counts))
    zone3_count_series['a1'].append(np.mean(a1_zone3_counts))
    zone3_count_series['a2'].append(np.mean(a2_zone3_counts))
    zone4_count_series['a1'].append(np.mean(a1_zone4_counts))
    zone4_count_series['a2'].append(np.mean(a2_zone4_counts))
    time_to_activation_series['a1'].append(a1_time_to_activation)
    time_to_activation_series['a2'].append(a2_time_to_activation)
    time_to_reward_series['a1'].append(a1_time_to_reward)
    time_to_reward_series['a2'].append(a2_time_to_reward)
    deviation_from_reward_series['a1'].append(a1_deviation_from_reward)
    deviation_from_reward_series['a2'].append(a2_deviation_from_reward)
    deviation_from_activation_series['a1'].append(a1_deviation_from_activation)
    deviation_from_activation_series['a2'].append(a2_deviation_from_activation)
        
    return (seq, 
            success_series,
            rewards_activated_series, 
            rewards_collected_series, 
            zone1_count_series,
            zone2_count_series,
            zone3_count_series,
            zone4_count_series, 
            time_to_activation_series,
            time_to_reward_series, 
            deviation_from_activation_series, 
            deviation_from_reward_series)


def plot_qval(agent, env_set):
    grid_size = env_set['grid_size']
    save_dir = env_set['save_dir']

    # Initialize matrices for Q-values and occupancy
    q1_mat = np.zeros((grid_size, grid_size, grid_size, grid_size, 7))
    q2_mat = np.zeros((grid_size, grid_size, grid_size, grid_size, 7))
    occu_mat = np.zeros((grid_size, grid_size, grid_size, grid_size))

    reward_place_to_coord = {
        () : 0,
        ('up','right') : 1,
        ('right','down') : 2,
        ('down','left') : 3,
        ('up','left') : 4,
        ('up','down') : 5,
        ('right','left') : 6,
    }

    # Populate Q-value and occupancy matrices
    for state in agent.q_values1.keys():
        agent1, agent2 = state
        a1_pos, rewards_active = agent1
        a2_pos, rewards_active = agent2
        a1_x, a1_y = a1_pos
        a2_x, a2_y = a2_pos
        idx = reward_place_to_coord[rewards_active]
        q1_mat[a1_x, a1_y, a2_x, a2_y, idx] = agent.q_values1[state]
        q2_mat[a1_x, a1_y, a2_x, a2_y, idx] = agent.q_values2[state]

    for state in agent.occupancy.keys():
        a1_pos, a2_pos = state
        a1_x, a1_y = a1_pos
        a2_x, a2_y = a2_pos
        occu_mat[a1_x, a1_y, a2_x, a2_y] = agent.occupancy[state]

    plt.figure(figsize=(grid_size * 1.4 * 7, grid_size * 5))

    for i in range(7):
        
        norm = plt.Normalize()
        
        def identity(x):
            return x
        def log_1(x):
            return np.log(x + 1)
        def log_3(x):
            return np.log(np.log(np.log(x + 1) + 1) + 1)
        
        func = identity
        if 0 < i:
            func = log_1
        if 4 < i:
            func = log_3
        
        # Plot for Q-value table 1, agent 1's x vs y
        plt.subplot(4, 7, i + 7 * 0 + 1)
        plt.imshow(func(q1_mat[:, :, :, :, i].sum(axis=(2, 3))), norm=norm)
        cb = plt.colorbar()
        cb.remove()
        plt.title('Agent1 Q-Table for Agent1', fontsize=24, fontweight='bold')
        plt.xlabel('X position', fontsize=16)
        plt.ylabel('Y position', fontsize=16)
        plt.yticks(range(grid_size), range(grid_size)[::-1])

        # Plot for Q-value table 1, agent 2's x vs y
        plt.subplot(4, 7, i + 7 * 1 + 1)
        plt.imshow(func(q1_mat[:, :, :, :, i].sum(axis=(0, 1))), norm=norm)
        cb = plt.colorbar()
        cb.remove()
        plt.title('Agent1 Q-Table for Agent2', fontsize=24, fontweight='bold')
        plt.xlabel('X position', fontsize=16)
        plt.ylabel('Y position', fontsize=16)
        plt.yticks(range(grid_size), range(grid_size)[::-1])

        # Plot for Q-value table 2, agent 1's x vs y
        plt.subplot(4, 7, i + 7 * 2 + 1)
        plt.imshow(func(q2_mat[:, :, :, :, i].sum(axis=(2, 3))), norm=norm)
        cb = plt.colorbar()
        cb.remove()
        plt.title('Agent2 Q-Table for Agent1', fontsize=24, fontweight='bold')
        plt.xlabel('X position', fontsize=16)
        plt.ylabel('Y position', fontsize=16)
        plt.yticks(range(grid_size), range(grid_size)[::-1])

        # Plot for Q-value table 2, agent 2's x vs y
        plt.subplot(4, 7, i + 7 * 3 + 1)
        plt.imshow(func(q2_mat[:, :, :, :, i].sum(axis=(0, 1))), norm=norm)
        cb = plt.colorbar()
        cb.remove()
        plt.title('Agent2 Q-Table for Agent2', fontsize=24, fontweight='bold')
        plt.xlabel('X position', fontsize=16)
        plt.ylabel('Y position', fontsize=16)
        plt.yticks(range(grid_size), range(grid_size)[::-1])


        # # Plot for occupancy matrix for Agent1
        # if i == 0:
        #     func = log_3
        #     plt.subplot(7, 5, 5 * i + 5)
        #     plt.imshow(func(occu_mat.sum(axis=(2, 3))), norm=norm)
        #     plt.colorbar()
        #     plt.title('Occupancy of Agent 1')
        #     plt.xlabel('x position')
        #     plt.ylabel('y position')
        
        # # Plot for occupancy matrix for Agent2
        # if i == 1:
        #     func = log_1
        #     plt.subplot(7, 5, 5 * i + 5)
        #     plt.imshow(func(occu_mat.sum(axis=(0, 1))), norm=norm)
        #     plt.colorbar()
        #     plt.title('Occupancy of Agent 2')
        #     plt.xlabel('x position')
        #     plt.ylabel('y position')

    plt.tight_layout()
    plt.savefig(save_dir + 'qval_mat.png')
    plt.close()


def plot_trajectory(traj1, traj2, grid_size=5):
    """
    Plot the trajectories of two agents on a 2D grid.

    Parameters:
    traj1 (list): List of coordinates representing the trajectory of agent 1.
    traj2 (list): List of coordinates representing the trajectory of agent 2.

    Returns:
    fig (matplotlib.figure.Figure): The generated figure object.
    """

    cmap = cm.get_cmap('viridis')
    rgb_values = cmap.colors
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True) 
    # Plot for traj1
    points = np.expand_dims(np.array(traj1), axis=1)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(traj1)-1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(traj1)))
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)  # Plot in the first subplot
    axs[0].set_xlim(-1, grid_size)
    axs[0].set_ylim(grid_size, -1)  # Change the y-axis limits to reverse the origin
    axs[0].scatter(traj1[0][0],traj1[0][1], color=rgb_values[0])
    axs[0].scatter([0,grid_size-1],[int(grid_size/2),int(grid_size/2)], color=rgb_values[-1])
    axs[0].set_title('agent1')
    # axs[0].text(-1,-1.5, traj1)
    fig.colorbar(line, ax=axs[0])
    # Plot for traj2
    points = np.expand_dims(np.array(traj2), axis=1)  # Use traj2 instead of traj1
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(traj2)-1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(traj2)))
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)  # Plot in the second subplot
    axs[1].set_xlim(-1, grid_size)
    axs[1].set_ylim(grid_size, -1)  # Change the y-axis limits to reverse the origin
    axs[1].scatter(traj2[0][0],traj2[0][1], color=rgb_values[0])
    axs[1].scatter([0,grid_size-1],[int(grid_size/2),int(grid_size/2)], color=rgb_values[-1])
    axs[1].set_title('agent2')
    # axs[1].text(-1,-1.5,traj2)
    fig.colorbar(line, ax=axs[1])
    
    return fig


def plot_predictor(agent,env_set):
    plt.subplot(2,1,1)
    plt.plot(agent.accuracy)
    plt.title('Accuracy')
    plt.subplot(2,1,2)
    plt.plot(agent.rp_length)
    plt.title('Samples used')
    plt.tight_layout()
    plt.savefig(env_set['save_dir']+'predictor_stats.png')
    plt.close()


def find_leader(traj1, traj2):
    if traj1 == traj2: return 0
    else:
        final_state = traj1[-1]
        distance1 = [np.sum(np.abs(np.array(ele) - np.array(final_state))) for ele in traj1]
        distance2 = [np.sum(np.abs(np.array(ele) - np.array(final_state))) for ele in traj2]
        idx = 1
        while idx < len(distance1):
            if distance1[idx] < distance2[idx]:
                return 1
            elif distance1[idx] > distance2[idx]:
                return 2 
            else:
                idx += 1
        return 0