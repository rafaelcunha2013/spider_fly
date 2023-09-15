from agent import Agent
from enviroment import SpiderFly2D, Fly, Spider
from utilities import convert, run_trained, plot_graphs3, train


import numpy as np
import matplotlib.pyplot as plt
import copy
import os

################################
### Main parameters ############
state_len = 6
fly1 = Fly(0, 'F1 ')
fly2 = Fly(state_len-1, 'F2 ')
flies = (fly1, fly2)
render = True
action_len = 3
eps_decay = 200000.
eps_min = 0.01
eps_max = 1.

iteractive = False
plt.ion() if iteractive else None # Turn iteractive mode on.
epochs = 50
algorithm = 'q-learning'
print_li = False

if algorithm == 'q-learning':
    ################################
    ### Environment parameters #####
    spiders = (Spider(round(state_len/2), 'Sp1'),)
    file_name = 'single4'
    env = SpiderFly2D(spiders, flies, render_mode=render, size=state_len, max_steps=300, name=file_name)
    agent1 = Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=0.5, gamma=0.99)
    train(epochs, env, (agent1, ), algorithm, iteractive=iteractive)
    plt.savefig(os.path.join('.', 'fig', 'single_agent5.png'))

if algorithm == 'iql':
    ################################
    ###### Double agent ############
    spiders = (Spider(round(state_len/2), 'Sp1'), Spider(round(state_len/2) + 1, 'Sp2'))
    file_name = 'double01'
    env = SpiderFly2D(spiders, flies, render_mode=render, size=state_len, max_steps=300, name=file_name)
    agents = (Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=0.5, gamma=0.99),
            Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=0.5, gamma=0.99))
    train(epochs, env, agents, "iql", iteractive=iteractive)
    plt.savefig(os.path.join('.', 'fig', 'double_agent4.png'))