from agent import Agent
from enviroment import SpiderFly2D, Fly, Spider
import numpy as np
import matplotlib.pyplot as plt
import copy
from utilities import convert


# It is probably not been used ###########################
def run_trained(agent, env_test):
    epochs = 1
    n_steps = []
    agent.epsilon = 0
    for _ in range(epochs):

        state, _ = env_test.reset()
        state = convert(state)
        if env_test.render_mode:
            env_test.render()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            
            action1 = agent.select_action(state)

            action = (action1,)
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            next_state = convert(next_state)

            state = next_state

            if env_test.render_mode:
                env_test.render()

        #n_steps.append(env.steps)

    return env_test.steps

fly1 = Fly(0, 'F1 ')
fly2 = Fly(11, 'F2 ')
spd1 = Spider(5, 'Sp1')