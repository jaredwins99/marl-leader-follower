import random
import numpy as np
import pygame  # Add this line to import pygame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class SuccessorRepresentationAgent:
    def __init__(self, color, target, width, height, draw_radius):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.color = color
        self.target = target
        self.successor_representation = np.zeros((width, height))
        self.width = width
        self.height = height
        self.draw_radius = draw_radius

    def get_state(self):
        return (self.x, self.y)

    def choose_action(self):
        possible_actions = ["up", "down", "left", "right"]
        action_probabilities = [self.successor_representation[self.take_action(a)] for a in possible_actions]
        chosen_action = np.random.choice(possible_actions, p=action_probabilities / np.sum(action_probabilities))
        return chosen_action

    def take_action(self, action):
        if action == "up":
            return max(0, self.x), max(0, self.y - 1)
        elif action == "down":
            return max(0, self.x), min(self.height, self.y + 1)
        elif action == "left":
            return max(0, self.x - 1), self.y
        elif action == "right":
            return min(self.width, self.x + 1), self.y

    def calculate_successor_representation(self, next_state, gamma):
        self.successor_representation *= gamma
        self.successor_representation[next_state] += 1

    def move_towards_target(self):
        if self.x < self.target[0]:
            self.x += 1
        elif self.x > self.target[0]:
            self.x -= 1

        if self.y < self.target[1]:
            self.y += 1
        elif self.y > self.target[1]:
            self.y -= 1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.draw_radius)  # Draw a single pixel for the agent
    

class QLearningAgent:
    def __init__(self, color, width, height, draw_radius, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.x = random.randint(0, width - 1)
        self.y = random.randint(0, height - 1)
        self.color = color
        self.width = width
        self.height = height
        self.draw_radius = draw_radius
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
        self.successor_representation = np.zeros((width, height))
        self.previous_state = None
        self.previous_action = None

    def get_state(self):
        return (self.x, self.y)

    def choose_action(self):
        possible_actions = ["up", "down", "left", "right"]
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)  # Explore: choose a random action
        else:
            q_values = [self.get_q_value(self.get_next_state(a)) for a in possible_actions]
            return possible_actions[np.argmax(q_values)]

    def get_next_state(self, action):
        if action == "up":
            return max(0, self.x), max(0, self.y - 1)
        elif action == "down":
            return max(0, self.x), min(self.height - 1, self.y + 1)
        elif action == "left":
            return max(0, self.x - 1), self.y
        elif action == "right":
            return min(self.width - 1, self.x + 1), self.y

    def get_q_value(self, state):
        return self.q_values.get(state, 0)

    def update_q_value(self, state, action, reward, next_state):
        max_next_q_value = max([self.get_q_value(self.get_next_state(a)) for a in ["up", "down", "left", "right"]])
        self.q_values[self.get_state()] = (1 - self.alpha) * self.get_q_value(self.get_state()) + \
                                                   self.alpha * (reward + self.gamma * max_next_q_value)

    # def calculate_successor_representation(self, next_state):
    #     self.successor_representation *= self.gamma
    #     self.successor_representation[next_state] += 1

    def move(self, action):
        if action == "up":
            self.y = max(0, self.y - 1)
        elif action == "down":
            self.y = min(self.height - 1, self.y + 1)
        elif action == "left":
            self.x = max(0, self.x - 1)
        elif action == "right":
            self.x = min(self.width - 1, self.x + 1)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.draw_radius)


class JointAgent:
    # Only works for two agents
    def __init__(self, width, height, draw_radius, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.x1 = random.randint(0, width - 1)
        self.y1 = random.randint(0, height - 1)
        self.x2 = random.randint(0, width - 1)
        self.y2 = random.randint(0, height - 1)
        self.width = width
        self.height = height
        # self.draw_radius = draw_radius # unneeded, too slow
        self.alpha = alpha # learning rate
        self.gamma = gamma # future discount
        self.epsilon = epsilon # epsilon greedy
        self.q_values1 = {} # this is actually the value function for each state
        self.q_values2 = {}
        self.occupancy = {}
        self.previous_state = None
        self.previous_action = None  
        self.possible_actions = ["stay","up", "down", "left", "right"]
        self.rewards_active = ()

    def update_occupancy(self):
        self.occupancy[(self.get_state(0)[0], self.get_state(1)[0])] = 1 + self.occupancy.get((self.get_state(0)[0], self.get_state(1)[0]),0)

    def reset(self):
        self.x1 = random.randint(0, self.width - 1)
        self.y1 = random.randint(0, self.height - 1)
        self.x2 = random.randint(0, self.width - 1)
        self.y2 = random.randint(0, self.height - 1)
        self.rewards_active = ()

    def reset_to_center(self):
        self.x1, self.y1 = (int(self.width/2),int(self.height/2))
        self.x2, self.y2 = (int(self.width/2),int(self.height/2))
        self.rewards_active = ()
    
    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.possible_actions), random.choice(self.possible_actions)
        else:
            # list all action pairs (e.g., 25)
            action_pairs = [(a1,a2) for a1 in self.possible_actions for a2 in self.possible_actions]
            
            # update
            q_values = [self.get_q_value(0,
                                         self.get_next_state(((self.x1, self.y1), self.rewards_active), a1),
                                         self.get_next_state(((self.x2, self.y2), self.rewards_active), a2)) 
                        for a1,a2 in action_pairs]
            
            # choose a best best action (random because of ties)
            max_indices = np.where(q_values == np.max(q_values))[0]
            idx = random.choice(max_indices)
            action1 = action_pairs[idx][0]
            
            # update
            q_values = [self.get_q_value(1,
                                         self.get_next_state(((self.x1, self.y1), self.rewards_active),a1),
                                         self.get_next_state(((self.x2, self.y2), self.rewards_active),a2)) 
                        for a1,a2 in action_pairs]
            
            # choose a best best action (random because of ties)
            max_indices = np.where(q_values == np.max(q_values))[0]
            idx = random.choice(max_indices)
            action2 = action_pairs[idx][1]
            
            return action1, action2  

    def get_next_state(self, states, action):
        
        pos, _ = states
        
        if action == "up":
            next_state = (max(0, pos[0]), max(0, pos[1] - 1)), self.rewards_active
        elif action == "down":
            next_state = (max(0, pos[0]), min(self.height - 1, pos[1] + 1)), self.rewards_active
        elif action == "left":
            next_state = (max(0, pos[0] - 1), pos[1]), self.rewards_active
        elif action == "right":
            next_state = (min(self.width - 1, pos[0] + 1), pos[1]), self.rewards_active
        elif action == "stay":
            next_state = (pos[0], pos[1]), self.rewards_active
        return next_state

    def get_q_value(self, idx, state1, state2):
        if idx == 0:
            return self.q_values1.get((state1,state2), 0)
        elif idx == 1:
            return self.q_values2.get((state1,state2), 0)

    def get_state(self,idx):
        if idx == 0:
            return (self.x1, self.y1), self.rewards_active
        elif idx == 1:
            return (self.x2, self.y2), self.rewards_active

    def update_q_value(self, state1, state2, reward1, reward2):
        # Gauss-Seidal Value Iteration
        next_state1 = [self.get_next_state(state1, a) for a in self.possible_actions]
        next_state2 = [self.get_next_state(state2, a) for a in self.possible_actions]
        
        current_state = (self.get_state(0), self.get_state(1))
        
        next_q_values = [self.get_q_value(0,state1,state2) for state1 in next_state1 for state2 in next_state2]
        self.q_values1[current_state] = (1 - self.alpha) * self.get_q_value(0,self.get_state(0),self.get_state(1)) + \
                                               self.alpha * (reward1 + self.gamma * max(next_q_values))
                                               
        next_q_values = [self.get_q_value(1,state1,state2) for state1 in next_state1 for state2 in next_state2]                                                                                     
        self.q_values2[current_state] = (1 - self.alpha) * self.get_q_value(1,self.get_state(0),self.get_state(1)) + \
                                               self.alpha * (reward2 + self.gamma * max(next_q_values))

    def move(self, idx, action):
        if idx == 0:
            if action == "up":
                self.y1 = max(0, self.y1 - 1)
            elif action == "down":
                self.y1 = min(self.height - 1, self.y1 + 1)
            elif action == "left":
                self.x1 = max(0, self.x1 - 1)
            elif action == "right":
                self.x1 = min(self.width - 1, self.x1 + 1)
        elif idx == 1:
            if action == "up":
                self.y2 = max(0, self.y2 - 1)
            elif action == "down":
                self.y2 = min(self.height - 1, self.y2 + 1)
            elif action == "left":
                self.x2 = max(0, self.x2 - 1)
            elif action == "right":
                self.x2 = min(self.width - 1, self.x2 + 1)


class JointAgent_Predictor(JointAgent):
    # Agent 1 predicts agent 2's action
    def __init__(self, width, height, draw_radius, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(width, height, draw_radius, alpha, gamma, epsilon)
        self.predictor = None
        self.predictor_built = False
        self.explore = False
        self.accuracy = []
        self.rp_length = []
        self.right_prediction = 0

    def set_predictor(self, predictor='logistic'):
        if predictor == 'logistic':
            self.predictor = LogisticRegression()
        else:
            raise NotImplementedError

    def build_predictor(self, input, output):
        if np.unique(output).shape[0] == 1:
            print("No need to build predictor")
            return 
        self.predictor.fit(input, output)
        y_pred = self.predictor.predict(input)
        self.accuracy.append(accuracy_score(output, y_pred))
        self.rp_length.append(input.shape[0])
        self.predictor_built = True
        # print("Training accuracy:", self.accuracy[-1])
    
    def predict_action(self):
        if self.predictor is not None:
            return self.predictor.predict(np.array(self.get_state(0)+ self.get_state(1)).reshape(1,-1))[0]
        else:
            raise NotImplementedError  

    # need to modify this two function
    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            self.explore = True
            return random.choice(self.possible_actions), random.choice(self.possible_actions)
        else:
            self.explore = False
            predicted_actions = [self.predict_action()] if self.predictor_built else self.possible_actions
            action_pairs = [(a1,a2) for a1 in self.possible_actions for a2 in predicted_actions]
            q_values = [self.get_q_value(0,self.get_next_state((self.x1,self.y1),a1),self.get_next_state((self.x2,self.y2),a2)) for a1,a2 in action_pairs]
            max_indices = np.where(q_values == np.max(q_values))[0]
            idx = random.choice(max_indices)
            action1 = action_pairs[idx][0]
            action_pairs = [(a1,a2) for a1 in self.possible_actions for a2 in self.possible_actions]
            q_values = [self.get_q_value(1,self.get_next_state((self.x1,self.y1),a1),self.get_next_state((self.x2,self.y2),a2)) for a1,a2 in action_pairs]
            max_indices = np.where(q_values == np.max(q_values))[0]
            idx = random.choice(max_indices)
            action2 = action_pairs[idx][1]
            if action2 in predicted_actions:
                self.right_prediction += 1
            return action1, action2  
        
    def update_q_value(self, state1,state2,reward1,reward2):
        next_state1 = [self.get_next_state(state1,a) for a in self.possible_actions]
        # predicted_actions = [self.predict_action()] if self.predictor_built else self.possible_actions
        # next_state2 = [self.get_next_state(state2,a) for a in predicted_actions]
        next_state2 = [self.get_next_state(state2,a) for a in self.possible_actions]
        next_q_values = [self.get_q_value(0,state1,state2) for state1 in next_state1 for state2 in next_state2]
        self.q_values1[(self.get_state(0),self.get_state(1))] = (1 - self.alpha) * self.get_q_value(0,self.get_state(0),self.get_state(1)) + \
                                               self.alpha * (reward1 + self.gamma * max(next_q_values))
        next_state2 = [self.get_next_state(state2,a) for a in self.possible_actions]
        next_q_values = [self.get_q_value(1,state1,state2) for state1 in next_state1 for state2 in next_state2]
        self.q_values2[(self.get_state(0),self.get_state(1))] = (1 - self.alpha) * self.get_q_value(1,self.get_state(0),self.get_state(1)) + \
                                               self.alpha * (reward2 + self.gamma * max(next_q_values))


class JointAgent_AbsolutePredictor(JointAgent):
    # Agent 1 knows agent 2's action exactly
    def __init__(self, width, height, draw_radius, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(width, height, draw_radius, alpha, gamma, epsilon)
        self.action2 = None

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            action1, action2 = random.choice(self.possible_actions), random.choice(self.possible_actions)
        else:
            action_pairs = [(a1,a2) for a1 in self.possible_actions for a2 in self.possible_actions]
            q_values = [self.get_q_value(1,self.get_next_state((self.x1,self.y1),a1),self.get_next_state((self.x2,self.y2),a2)) for a1,a2 in action_pairs]
            max_indices = np.where(q_values == np.max(q_values))[0]
            idx = random.choice(max_indices)
            action2 = action_pairs[idx][1]
            action_pairs = [(a1,a2) for a1 in self.possible_actions for a2 in [action2]]
            q_values = [self.get_q_value(0,self.get_next_state((self.x1,self.y1),a1),self.get_next_state((self.x2,self.y2),a2)) for a1,a2 in action_pairs]
            max_indices = np.where(q_values == np.max(q_values))[0]
            idx = random.choice(max_indices)
            action1 = action_pairs[idx][0]
        self.action2 = action2
        return action1, action2  
        
    def update_q_value(self, state1, state2, reward1, reward2):
        next_state1 = [self.get_next_state(state1,a) for a in self.possible_actions]
        # predicted_actions = [self.action2]
        # next_state2 = [self.get_next_state(state2,a) for a in predicted_actions]
        next_state2 = [self.get_next_state(state2,a) for a in self.possible_actions]
        next_q_values = [self.get_q_value(0,state1,state2) for state1 in next_state1 for state2 in next_state2]
        self.q_values1[(self.get_state(0),self.get_state(1))] = (1 - self.alpha) * self.get_q_value(0,self.get_state(0),self.get_state(1)) + \
                                               self.alpha * (reward1 + self.gamma * max(next_q_values))
        next_state2 = [self.get_next_state(state2,a) for a in self.possible_actions]
        next_q_values = [self.get_q_value(1,state1,state2) for state1 in next_state1 for state2 in next_state2]
        self.q_values2[(self.get_state(0),self.get_state(1))] = (1 - self.alpha) * self.get_q_value(1,self.get_state(0),self.get_state(1)) + \
                                               self.alpha * (reward2 + self.gamma * max(next_q_values))