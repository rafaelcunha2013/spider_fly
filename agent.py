import numpy as np
import math
import random

class Agent:

    def __init__(self, state_len, action_len, eps_decay, eps_min, eps_max, lr=1e-3, gamma=0.99):
        self.state_len = state_len
        self.action_len = action_len
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.lr = lr
        self.gamma = gamma

        self.epsilon = eps_max
        self.episode = 0
        # self.q = np.zeros((state_len * 3 - 4, action_len), dtype=float) # Memmory eficient version
        self.q = np.zeros((state_len * 3, action_len), dtype=float)

    def select_action(self, state):

        if random.random() > self.epsilon:
            # Use numpy's argmax function to find indices of maximum values
            arr = self.q[state]
            max_indices = np.argwhere(arr == np.amax(arr)).flatten()
            # Select one index randomly among max_indices
            action = np.random.choice(max_indices)-1
            # action = np.argmax(self.q[state])-1
        else:
            action = np.random.choice([-1, 0, 1], size=1)

        return action
    

    def update_q(self, transitions):
        state, action, reward, next_state, done = transitions

        #if reward:
            #print(reward)
        if done:
            self.q[state, action] += self.lr * (reward - self.q[state, action])
        else:
            self.q[state, action] += self.lr * (reward + self.gamma * np.max(self.q[next_state]) - self.q[state, action])

    def update_epsilon(self):
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-self.episode / self.eps_decay)
        self.episode += 1


if __name__ == '__main__':
    state_len = 4
    action_len = 3
    eps_decay = 100.
    eps_min = 0.01
    eps_max = 1.
    agent1 = Agent(state_len, action_len, eps_decay, eps_min, eps_max)
    agent2 = Agent(state_len, action_len, eps_decay, eps_min, eps_max)




