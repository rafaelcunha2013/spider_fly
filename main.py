from agent import Agent
from enviroment import SpiderFly2D, Fly, Spider
from utilities import convert, run_trained, plot_graphs, train


import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from datetime import datetime
import time

start_time = time.time()
################################
### Main parameters ############
state_len = 6
action_len = 3
eps_decay = 200000.
eps_min = 0.01
eps_max = 1.
lr = 1e-3
gamma = 0.9
max_steps = 300

render = True
iteractive = False
print_li = False
epochs = 500
algorithm = 'q-learning'


#####################################################
fly1 = Fly(0, 'F1 ')
fly2 = Fly(state_len-1, 'F2 ')
flies = (fly1, fly2)
plt.ion() if iteractive else None # Turn iteractive mode on.
unique_id = datetime.now().strftime("%m_%d__%H_%M_%S__%f")[:-4]
file_name = f'{algorithm}_{unique_id}_' 

if algorithm == 'q-learning':
    spiders = (Spider(round(state_len/2), 'Sp1'), )
    agents = (Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=lr, gamma=gamma), )

if algorithm == 'iql':
    ################################
    ###### Double agent ############
    spiders = (Spider(round(state_len/2), 'Sp1'), Spider(round(state_len/2) + 1, 'Sp2'))
    agents = (Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=lr, gamma=gamma),
            Agent(state_len, action_len, eps_decay, eps_min, eps_max, lr=lr, gamma=gamma))
    
if algorithm == 'iql_fo':
    spiders = (Spider(round(state_len/2), 'Sp1'), Spider(round(state_len/2) + 1, 'Sp2'))
    agents = (Agent(state_len ** 2, action_len, eps_decay, eps_min, eps_max, lr=lr, gamma=gamma),
            Agent(state_len ** 2, action_len, eps_decay, eps_min, eps_max, lr=lr, gamma=gamma))

env = SpiderFly2D(spiders, flies, render_mode=render, size=state_len, max_steps=max_steps, name=file_name, print_li=print_li)
train(epochs, env, agents, algorithm, iteractive=iteractive)

end_time = time.time()
elapsed_time = round((end_time - start_time)/60, 2)
print(f"Elapsed time: {elapsed_time} minutes")
