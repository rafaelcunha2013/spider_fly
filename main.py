from agent import Agent
from enviroment import SpiderFly2D, Fly, Spider
from utilities import convert, run_trained, plot_graphs3, train


import numpy as np
import matplotlib.pyplot as plt
import copy

################################
### Environment parameters #####
length = 6
fly1 = Fly(0, 'F1 ')
fly2 = Fly(length-1, 'F2 ')
spd1 = Spider(round(length/2), 'Sp1')
# spd2 = Spider(6, 'Sp2')
#spiders = (spd1, spd2)
spiders = (spd1,)
flies = (fly1, fly2)
render = True
file_name = 'single3'
env = SpiderFly2D(spiders, flies, render_mode=render, size=length, max_steps=300, name=file_name)
#####################################
### Agent parameters ################
state_len = length
action_len = 3
eps_decay = 200000.
eps_min = 0.01
eps_max = 1.
agent1 = Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=0.5, gamma=0.99)
# agent2 = Agent(state_len, action_len, eps_decay, eps_min, eps_max)

plt.ion() # Turn interactive mode on.
epochs = 50
train(epochs, env, (agent1, ), "q-learning")
plt.savefig('single_agent1.png')


################################
###### Double agent ############
spiders = (Spider(round(length/2), 'Sp1'), Spider(round(length/2) + 1, 'Sp2'))
file_name = 'double00'
env = SpiderFly2D(spiders, flies, render_mode=render, size=length, max_steps=300, name=file_name)
agents = (Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=0.5, gamma=0.99),
          Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=0.5, gamma=0.99))
train(epochs, env, agents, "iql")
plt.savefig('double_agent1.png')