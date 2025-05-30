# source activate py3.8

import pygame
import os,sys,subprocess
from agents import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FPS = 120

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 200,200
NUM_FRAMES = 300  # Adjust as needed
draw_radius = 3
target = (WIDTH // 2, HEIGHT // 2)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cooperative Successor Representation Agent Task")
clock = pygame.time.Clock()

# Create agents and target
agent1 = SuccessorRepresentationAgent(RED, target, WIDTH, HEIGHT, draw_radius)
agent2 = SuccessorRepresentationAgent(BLUE, target, WIDTH, HEIGHT, draw_radius)

save_dir = 'test/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Main loop
running = True
frame_count = 0
while running and frame_count < NUM_FRAMES:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Move agents towards the target cooperatively
    agent1.move_towards_target()
    agent2.move_towards_target()
    # Check if agents have reached the target exactly
    if agent1.get_state() == target and agent2.get_state() == target:
        print("Both agents have reached the target!")
        running = False  # Stop the simulation
    # Update successor representations based on individual movements
    next_state1 = (agent1.x, agent1.y)
    next_state2 = (agent2.x, agent2.y) 
    agent1.calculate_successor_representation(next_state1, gamma=0.9)
    agent2.calculate_successor_representation(next_state2, gamma=0.9)
    # Draw agents and target
    screen.fill(WHITE)
    pygame.draw.circle(screen, GREEN, target, draw_radius) 
    # Draw border lines
    pygame.draw.line(screen, BLACK, (0, 0), (WIDTH - 1, 0), 1)  # Top border
    pygame.draw.line(screen, BLACK, (0, 0), (0, HEIGHT - 1), 1)  # Left border
    pygame.draw.line(screen, BLACK, (WIDTH - 1, 0), (WIDTH - 1, HEIGHT), 2)  # Right border
    pygame.draw.line(screen, BLACK, (0, HEIGHT - 1), (WIDTH, HEIGHT - 1), 2)  # Bottom border
    agent1.draw(screen)
    agent2.draw(screen)
    # Convert the Pygame surface to a NumPy array
    pixels = pygame.surfarray.array3d(screen)
    # Save frame as an image
    pygame.image.save(pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2))), save_dir+f"frame_{frame_count:03d}.png")
    # Update display
    pygame.display.flip()
    # Cap the frame rate
    clock.tick(FPS)
    frame_count += 1

# Quit Pygame
pygame.quit()
# Generate a movie and remove the *png
command = 'ffmpeg -framerate 60 -i {}frame_%03d.png -c:v libx264 -pix_fmt yuv420p {}output.mp4'.format(save_dir,save_dir)
results = subprocess.run(command,shell=True, check=True, stdout=subprocess.PIPE, text=True)
command = 'rm {}*.png'.format(save_dir)
results = subprocess.run(command,shell=True, check=True, stdout=subprocess.PIPE, text=True)



## One RL agent with Bellman Equation
save_dir = 'SingleRL/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Environment setup
pygame.init()
WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Foraging Agent in Open Arena")

# Environment setup
grid_size = 5
cell_width = WIDTH // grid_size
cell_height = HEIGHT // grid_size
reward_position = (2,2)  # Reward position

# Reward value
reward_collect = 10

# Initialize Q-learning agent with successor representation
ep = 0.6
agent = QLearningAgent(RED, grid_size, grid_size, draw_radius=20, gamma=0.98, epsilon=ep)

with open('{}q_value_ep{}.pkl'.format(save_dir,ep), 'rb') as f:
    q_values = pickle.load(f)

agent.q_values = q_values

# Pygame visualization
clock = pygame.time.Clock()
running = True
NUM_FRAMES = 1000
INIT_FRAME = 500
frame_save = 1
consecutive_iterations_to_check = 10  # Number of consecutive iterations to check for convergence
consecutive_iterations_below_threshold = 0
convergence_threshold = 0.001

frame_count = 0
while running and frame_count < NUM_FRAMES:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Agent chooses action
    action = agent.choose_action()
    next_state = agent.get_next_state(action)
    # Calculate reward based on the next state
    if next_state == reward_position:
        reward = reward_collect
    else:
        reward = 0  # Default reward for moving without reaching the reward
    # Update Q-value with successor representation
    if agent.previous_state is not None:
        agent.update_q_value(agent.previous_state, agent.previous_action, reward, next_state)
        agent.calculate_successor_representation(next_state)
    # Move the agent
    agent.move(action)
    # Check convergence
    if np.sum([val for key,val in agent.q_values.items()]) > 0 and frame_count > INIT_FRAME:
        res = {}
        for key in agent.q_values.keys():
            if key in agent.previous_q_values.keys():
                res[key] = agent.q_values[key] - agent.previous_q_values[key]
            else:
                res[key] = agent.q_values[key]
        max_q_change = np.max([val for key,val in res.items()])
        if max_q_change < convergence_threshold:
            consecutive_iterations_below_threshold += 1
        else:
            consecutive_iterations_below_threshold = 0
        if consecutive_iterations_below_threshold >= consecutive_iterations_to_check:
            print("Converged. Stopping training.")
            break
    if frame_count % frame_save == 0:
        screen.fill(WHITE)
        # Draw the heatmap background based on the state values
        if agent.q_values:
            max_q = np.max([val for key,val in agent.q_values.items()])
            if max_q >0:
                heatmap_surface = pygame.Surface((WIDTH, HEIGHT))
                heatmap_pixels = pygame.PixelArray(heatmap_surface)
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        x_gd = int(x / WIDTH * grid_size)
                        y_gd = int(y / HEIGHT * grid_size)
                        normalized_value = int(255 * (agent.get_q_value((x_gd,y_gd)) / max_q))
                        color = pygame.Color(255 - normalized_value, 255, 255 - normalized_value)
                        heatmap_pixels[x, y] = color
                del heatmap_pixels
                screen.blit(heatmap_surface, (0, 0))
        for i in range(grid_size):
            for j in range(grid_size):
                pygame.draw.rect(screen, GREEN, (j * cell_width, i * cell_height, cell_width, cell_height), 1)
        pygame.draw.circle(screen, RED, ((agent.x+0.5) * cell_width, (agent.y+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, BLUE, ((reward_position[0]+0.5) * cell_width, (reward_position[1]+0.5) * cell_height), agent.draw_radius)
        pixels = pygame.surfarray.array3d(screen)
        pygame.image.save(pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2))), save_dir+f"frame_{int(frame_count/frame_save):03d}.png")
        pygame.display.flip()
        clock.tick(FPS)  # Adjust the speed of the simulation
    # Update previous state and action for the next iteration
    agent.previous_state = agent.get_state()
    agent.previous_action = action
    agent.previous_q_values = agent.q_values.copy()
    frame_count += 1


# Quit Pygame
pygame.quit()

# Generate a movie and remove the *png
command = 'ffmpeg -framerate 10 -i {}frame_%03d.png -c:v libx264 -pix_fmt yuv420p {}output_ep{}.mp4'.format(save_dir,save_dir,ep)
command = 'ffmpeg -framerate 10 -i {}frame_%03d.png -c:v libx264 -pix_fmt yuv420p {}output_continued_ep{}.mp4'.format(save_dir,save_dir,ep)
results = subprocess.run(command,shell=True, check=True, stdout=subprocess.PIPE, text=True)
command = 'rm {}frame_*.png'.format(save_dir)
results = subprocess.run(command,shell=True, check=True, stdout=subprocess.PIPE, text=True)

with open('{}q_value_ep{}.pkl'.format(save_dir,ep), 'wb') as f:
    pickle.dump(agent.q_values, f)

# Use the learnt policy to do the foraging task and plot the successor representation at each location
with open('{}q_value_ep{}.pkl'.format(save_dir,ep), 'rb') as f:
    q_values = pickle.load(f)

possible_actions = ["up", "down", "left", "right"]
agent = QLearningAgent(RED, grid_size, grid_size, draw_radius=20, gamma=0.98, epsilon=ep)
agent.q_values = q_values
T = np.zeros((grid_size*grid_size, grid_size*grid_size)) # T(s',s) = P(s'|s)
for i in range(grid_size):
    for j in range(grid_size):
        agent.x = i
        agent.y = j
        s_idx = i*grid_size + j
        for a in possible_actions:
            s_tp1 = agent.get_next_state(a)
            T[s_tp1[0]*grid_size+s_tp1[1],i*grid_size+j] = agent.get_q_value(s_tp1)
        T[:,i*grid_size+j] = T[:,i*grid_size+j] / np.sum(T[:,i*grid_size+j])


SR = np.linalg.inv((np.identity(grid_size*grid_size) - agent.gamma * T.T))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(T); plt.colorbar()
plt.title('Transition Matrix')
plt.subplot(1,2,2)
plt.imshow(SR); plt.colorbar()
plt.title('SR')
plt.tight_layout()
plt.savefig(save_dir+'T_SR.png')

pygame.init()
WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Policy Evaluation with Successor Representation")

agent.successor_representation = SR
running = True
frame_count = 0

clock = pygame.time.Clock()
for i in range(grid_size):
    agent.x = i
    for j in range(grid_size):
        agent.y = j
        screen.fill(WHITE)
        heatmap_surface = pygame.Surface((WIDTH, HEIGHT))
        heatmap_pixels = pygame.PixelArray(heatmap_surface)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                x_gd = int(x / WIDTH * grid_size)
                y_gd = int(y / HEIGHT * grid_size)
                sr_vec = agent.successor_representation[agent.x*grid_size+agent.y,:]
                normalized_value = int(255*sr_vec[x_gd*grid_size+y_gd]/np.max(sr_vec))
                color = pygame.Color(255 - normalized_value, 255, 255 - normalized_value)
                heatmap_pixels[x, y] = color
        del heatmap_pixels
        screen.blit(heatmap_surface, (0, 0))
        for i in range(grid_size):
            for j in range(grid_size):
                pygame.draw.rect(screen, GREEN, (j * cell_width, i * cell_height, cell_width, cell_height), 1)
        pygame.draw.circle(screen, RED, ((agent.x+0.5) * cell_width, (agent.y+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, BLUE, ((reward_position[0]+0.5) * cell_width, (reward_position[1]+0.5) * cell_height), agent.draw_radius)
        pixels = pygame.surfarray.array3d(screen)
        pygame.image.save(pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2))), save_dir+f"frame_{int(frame_count):03d}.png")
        pygame.display.flip()
        clock.tick(FPS)  # Adjust the speed of the simulation
        frame_count += 1


pygame.quit()

'''
Multi-agent Q learning
'''
RED1 = (179,0,0)
RED2 = (127,0,0)

ep = 0.5
WIDTH, HEIGHT = 500, 500
grid_size = 10
cell_width = WIDTH // grid_size
cell_height = HEIGHT // grid_size
# reward_position1 = (0,2)  # Reward position
# reward_position2 = (4,2)
# reward_position2 = (0,2)
reward_position1 = (0,4)
reward_position2 = (9,4)
reward_collect = 10
reward_together = 10

agent = JointAgent(grid_size, grid_size, draw_radius=20, alpha=0.1, gamma=0.9, epsilon=ep)
# # test functions
# agent.choose_action()
# agent.get_state(0)
# agent.get_next_state(agent.get_state(0),'up')
# agent.get_q_value(0,agent.get_state(0),agent.get_state(1))
# agent.update_q_value(agent.get_state(0),agent.get_state(1),10,0)
# agent.q_values1
# agent.q_values2

# # Add priors
for i in range(grid_size):
    for j in range(grid_size):  
        agent.q_values1[(reward_position1,(i,j))] = 1
        agent.q_values2[((i,j),reward_position1)] = 1
        agent.q_values1[(reward_position2,(i,j))] = 1
        agent.q_values2[((i,j),reward_position2)] = 1


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()
running = True
NUM_FRAMES = 50000
INIT_FRAME = 500
frame_save = 1
consecutive_iterations_to_check = 1000  # Number of consecutive iterations to check for convergence
consecutive_iterations_below_threshold = 0
convergence_threshold = 0.1

# save_dir = 'MARL/'
# save_dir = 'MARL/MARL_leader/'
# save_dir = 'MARL/together/'
save_dir = 'MARL/DifferentReward/'

frame_count = 0
count1 = 0
count2 = 0
count3 = 0
consecutive_noreward = 0
reset = False
occupancy = {}
while running and frame_count < NUM_FRAMES:
    occupancy[(agent.get_state(0), agent.get_state(1))] = 1 + occupancy.get((agent.get_state(0), agent.get_state(1)),0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Agent chooses action
    action1, action2 = agent.choose_action()
    next_state1 = agent.get_next_state(agent.get_state(0),action1)
    next_state2 = agent.get_next_state(agent.get_state(1),action2)
    # Calculate reward based on the next state
    if (next_state1 == reward_position1 and next_state2 == reward_position1):
        # reward1, reward2 = reward_collect, reward_collect
        reward1, reward2 = reward_collect, reward_collect/2
        # reward1, reward2 = reward_collect, 0
        count1 += 1
        reset = True
    elif (next_state1 == reward_position2 and next_state2 == reward_position2):
        # reward1, reward2 = reward_collect, reward_collect
        reward1, reward2 = reward_collect, reward_collect/2
        # reward1, reward2 = reward_collect, 0
        count2 += 1
        reset = True
    # elif next_state1 == next_state2:
    #     reward1, reward2 = 0, reward_together
    #     # reward1, reward2 = reward_together, reward_together
    #     count3 += 1
    #     print('Hit reward3')
    else:
        reward1, reward2 = 0, 0  # Default reward for moving without reaching the reward
        consecutive_noreward += 1
    # Update Q-value with Bellman Equation
    if agent.previous_state is not None:
        agent.update_q_value(agent.get_state(0),agent.get_state(1),reward1,reward2)
    if consecutive_noreward > 100:
        reset = True
        consecutive_noreward = 0
    # Move the agent
    if reset:
        agent.reset()
        reset = False
    else:
        agent.move(0, action1)
        agent.move(1, action2)
    # # Check convergence
    # if np.sum([val for key,val in agent.q_values2.items()]) > 0 and frame_count > INIT_FRAME:
    #     res = {}
    #     for key in agent.q_values2.keys():
    #         res[key] = agent.q_values2[key] - agent.previous_q_values.get(key,0)
    #     max_q_change = np.max([val for key,val in res.items()])
    #     if max_q_change < convergence_threshold:
    #         consecutive_iterations_below_threshold += 1
    #     else:
    #         consecutive_iterations_below_threshold = 0
    #     if consecutive_iterations_below_threshold >= consecutive_iterations_to_check:
    #         print("Converged. Stopping training.")
    #         break
    if frame_count % frame_save == 0:
        screen.fill(WHITE)
        # display q values as text
        font = pygame.font.Font(None, 36)
        message = "Q1 value = {:.2f}".format(agent.get_q_value(0,agent.get_state(0),agent.get_state(1)))
        text_surface = font.render(message, True, (0,0,0))
        screen.blit(text_surface, (200,200))
        for i in range(grid_size):
            for j in range(grid_size):
                pygame.draw.rect(screen, GREEN, (j * cell_width, i * cell_height, cell_width, cell_height), 1)
        pygame.draw.circle(screen, RED1, ((agent.x1+0.5) * cell_width, (agent.y1+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, RED2, ((agent.x2+0.5) * cell_width, (agent.y2+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, BLUE, ((reward_position1[0]+0.5) * cell_width, (reward_position1[1]+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, BLUE, ((reward_position2[0]+0.5) * cell_width, (reward_position2[1]+0.5) * cell_height), agent.draw_radius)
        pixels = pygame.surfarray.array3d(screen)
        pygame.display.flip()
        # pygame.image.save(pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2))), save_dir+f"frame_{int(frame_count/frame_save):03d}.png")
        clock.tick(FPS)  # Adjust the speed of the simulation
    # Update previous state and action for the next iteration
    agent.previous_state = (agent.get_state(0), agent.get_state(1))
    agent.previous_action = (action1,action2)
    agent.previous_q_values = agent.q_values2.copy()
    frame_count += 1


pygame.quit()


with open(save_dir+'q_val1.pkl', 'wb') as f:
    pickle.dump(agent.q_values1, f)

with open(save_dir+'q_val2.pkl', 'wb') as f:
    pickle.dump(agent.q_values2, f)


# # Load the trained agent
# with open(save_dir+'q_val1.pkl', 'rb') as f:
#     qval1 = pickle.load(f)

# with open(save_dir+'q_val2.pkl', 'rb') as f:
#     qval2 = pickle.load(f)

# agent = JointAgent(grid_size, grid_size, draw_radius=20, alpha=0.1, gamma=0.9, epsilon=ep)
# agent.q_values1 = qval1
# agent.q_values2 = qval2

q1_mat = np.zeros((grid_size*grid_size,grid_size*grid_size))
q2_mat = np.zeros((grid_size*grid_size,grid_size*grid_size))
for key in agent.q_values1.keys():
    q1_mat[key[0][0]*grid_size+key[0][1], key[1][0]*grid_size+key[1][1]] = agent.q_values1[key]

for key in agent.q_values2.keys():
    q2_mat[key[0][0]*grid_size+key[0][1], key[1][0]*grid_size+key[1][1]] = agent.q_values2[key]

occu_mat = np.zeros((grid_size*grid_size, grid_size*grid_size))
for key in occupancy.keys():
    occu_mat[key[0][0]*grid_size+key[0][1], key[1][0]*grid_size+key[1][1]] = occupancy[key]



plt.figure(figsize=(18,8))
plt.subplot(1,3,1)
plt.imshow(q1_mat)
plt.colorbar()
plt.xlabel('State2'); plt.ylabel('State1'); plt.title('Q value 1')
plt.xticks(np.arange(grid_size*grid_size))
plt.gca().set_xticklabels([(i,j) for i in range(grid_size) for j in range(grid_size)], rotation=90)
plt.yticks(np.arange(grid_size*grid_size))
plt.gca().set_yticklabels([(i,j) for i in range(grid_size) for j in range(grid_size)])
plt.subplot(1,3,2)
plt.imshow(q2_mat)
plt.colorbar()
plt.xlabel('State2'); plt.ylabel('State1'); plt.title('Q value 2')
plt.xticks(np.arange(grid_size*grid_size))
plt.gca().set_xticklabels([(i,j) for i in range(grid_size) for j in range(grid_size)], rotation=90)
plt.yticks(np.arange(grid_size*grid_size))
plt.gca().set_yticklabels([(i,j) for i in range(grid_size) for j in range(grid_size)])
plt.subplot(1,3,3)
plt.imshow(occu_mat)
# plt.imshow(q1_mat / q2_mat)
plt.colorbar()
plt.xlabel('State2'); plt.ylabel('State1'); 
plt.title('occupancy')
plt.tight_layout()
plt.savefig(save_dir+'qval_mat.png')
plt.close()


# plot exploration after learning
agent.x1, agent.y1 = (grid_size/2-1,grid_size/2-1)
agent.x2, agent.y2 = (grid_size/2-1,grid_size/2-1)
agent.epsilon = 0.2
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()
running = True
frame_save = 1
frame_count = 0
while running and frame_count < NUM_FRAMES:
    if frame_count % frame_save == 0:
        screen.fill(WHITE)
        # display q values as text
        font = pygame.font.Font(None, 36)
        message = "Q1 value = {:.2f}".format(agent.get_q_value(0,agent.get_state(0),agent.get_state(1)))
        text_surface = font.render(message, True, (0,0,0))
        flipped = pygame.transform.flip(text_surface, False, True)
        screen.blit(pygame.transform.rotate(flipped, -90), (200,200))
        for i in range(grid_size):
            for j in range(grid_size):
                pygame.draw.rect(screen, GREEN, (j * cell_width, i * cell_height, cell_width, cell_height), 1)
        pygame.draw.circle(screen, RED1, ((agent.x1+0.5) * cell_width, (agent.y1+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, RED2, ((agent.x2+0.5) * cell_width, (agent.y2+0.5) * cell_height), agent.draw_radius)
        pygame.draw.circle(screen, BLUE, ((reward_position1[0]+0.5) * cell_width, (reward_position1[1]+0.5) * cell_height), agent.draw_radius/2)
        pygame.draw.circle(screen, BLUE, ((reward_position2[0]+0.5) * cell_width, (reward_position2[1]+0.5) * cell_height), agent.draw_radius/2)
        pixels = pygame.surfarray.array3d(screen)
        pygame.image.save(pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2))), save_dir+f"eval_frame_{int(frame_count/frame_save):03d}.png")
        pygame.display.flip()
    # Agent chooses action
    action1, action2 = agent.choose_action()
    agent.move(0, action1)
    agent.move(1, action2)
    # Calculate reward based on the next state
    if (agent.get_state(0) == reward_position1 and agent.get_state(1) == reward_position1) or (agent.get_state(0) == reward_position2 and agent.get_state(1) == reward_position2):
        running = False
    clock.tick(1)  # Adjust the speed of the simulation
    # Update previous state and action for the next iteration
    agent.previous_state = (agent.get_state(0), agent.get_state(1))
    agent.previous_action = (action1,action2)
    agent.previous_q_values = agent.q_values1.copy()
    frame_count += 1


pygame.quit()

command = 'ffmpeg -framerate 1 -i {}eval_frame_%03d.png -c:v libx264 -pix_fmt yuv420p {}eval.mp4'.format(save_dir,save_dir)
results = subprocess.run(command,shell=True, check=True, stdout=subprocess.PIPE, text=True)
