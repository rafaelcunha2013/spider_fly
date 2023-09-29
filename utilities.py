import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from agent import Agent

def plot_graphs(n_steps, n_eps, n_steps_trained):
    plt.figure(1)
    plt.plot(n_steps)
    plt.figure(2)
    plt.plot(n_eps)
    plt.figure(3)
    plt.plot(n_steps_trained)
    plt.draw()
    plt.pause(0.01)
    # print(agent1.q)


def plot_graphs2(n_steps, n_eps, n_steps_trained, first):
    # Create a figure and a set of subplots
    # if first:
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].plot(n_steps)
    axs[1].plot(n_steps_trained)
    axs[2].plot(n_eps)

    plt.draw()
    plt.pause(0.01)
    # print(agent1.q)

def plot_graphs3(n_steps, n_eps, n_steps_trained, lines, iteractive=False):
    lines[0].set_data(np.arange(len(n_steps)), n_steps)
    lines[1].set_data(np.arange(len(n_steps_trained)), n_steps_trained)
    lines[2].set_data(np.arange(len(n_eps)), n_eps)

    for line in lines:
        line.axes.relim()
        line.axes.autoscale_view()

    plt.draw()
    plt.pause(0.01) if iteractive else False

    return lines

def convert(state, env_length, env):
    fly_pos = env.fly_dict[str(state[0:2])]
    if fly_pos == 3:
        return 3 * env_length
    spider_pos = state[2]
    return int(fly_pos * env_length + spider_pos)

def convert_new(state, env_length, env):
    fly_pos = env.fly_dict[str(state[0:2])]
    if fly_pos == 3:
        return 3 * env_length
    spider_pos = state[2]
    if fly_pos == 0:
        converted_state = int(spider_pos - 1)
    else:
        converted_state = int(fly_pos * (env_length - 2) + spider_pos)
    return converted_state

def run_trained(agent, env_test, algorithm):
    epochs = 1
    n_steps = []
    for _ in range(epochs):

        state, _ = env_test.reset()
        if env_test.render_mode:
            env_test.render()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            
            action = compute_action(state, agent, algorithm, env_test)
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            state = next_state.copy()

            if env_test.render_mode:
                env_test.render()


    return env_test.steps

def compute_action(state, agents, algorithm, env):
    # return [agent.select_action(state) for agent in agents]
    if algorithm == "q-learning":
        state = convert(state, agents[0].state_len, env)
        return (agents[0].select_action(state), )
    
    if algorithm == "iql":
        flies_position = state[:2]
        actions = []
        for agent, spider in zip(agents, env.spiders):
            state = np.append(flies_position, spider.position)
            state = convert(state, agents[0].state_len, env)
            actions.append(agent.select_action(state))

        return actions

    if algorithm == "iql_fo":
        flies_position = state[:2]
        unified_spider_position = state[2] + state[3] * env.size
        actions = []
        for agent in agents:
            state = np.append(flies_position, unified_spider_position)
            state = convert(state, agents[0].state_len, env)
            actions.append(agent.select_action(state))

        return actions
    
def compute_transition(state, actions, reward, next_state, terminated, algorithm, agents, env):
    if algorithm == "q-learning":
        state = convert(state, agents[0].state_len, env)
        next_state = convert(next_state, agents[0].state_len, env)
        transitions = state, int(actions[0] + 1), reward, next_state, terminated
        return (transitions,)

    if algorithm == "iql":
        flies_position_st = state[:2]
        flies_position_next_st = next_state[:2]
        transitions = []
        for spider, action in zip(env.spiders, actions):
            state = np.append(flies_position_st, spider.position)
            state = convert(state, agents[0].state_len, env)
            next_state = np.append(flies_position_next_st, spider.position)
            next_state = convert(next_state, agents[0].state_len, env)

            transition = state, int(action + 1), reward, next_state, terminated
            transitions.append(transition)
        # transition2 = state, int(action[1] + 1), reward, next_state, terminated
        return transitions #(transition1, transition2)
    
    if algorithm == "iql_fo":
        flies_position_st = state[:2]
        unified_spider_position = state[2] + state[3] * env.size
        flies_position_next_st = next_state[:2]
        next_unified_spider_position = next_state[2] + next_state[3] * env.size
        
        state = np.append(flies_position_st, unified_spider_position)
        state = convert(state, agents[0].state_len, env)
        next_state = np.append(flies_position_next_st, next_unified_spider_position)
        next_state = convert(next_state, agents[0].state_len, env)
        transitions = []

        for action in actions:
            transition = state, int(action + 1), reward, next_state, terminated
            transitions.append(transition)

        return transitions

def train(epochs, env, agents, algorithm, iteractive=False):
    n_steps = []
    n_eps = []
    n_steps_trained = []

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].set_title('e-greedy number of steps')
    axs[1].set_title('greedy number of steps')
    axs[2].set_title('Current epsilon')
    lines = [ax.plot([], [])[0] for ax in axs]

    env_test = copy.deepcopy(env)
    env_test.name = env.name + 'trained'
    env_test.max_steps = 15

    agents_eval = [Agent(agent.state_len, agent.action_len, agent.eps_decay, eps_min=0., eps_max=0.) for agent in agents]

    for _ in range(epochs):
        state, _ = env.reset()

        if env.render_mode:
            env.render()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            
            action = compute_action(state, agents, algorithm, env)
            next_state, reward, terminated, truncated, _ = copy.deepcopy(env.step(action))
            transitions = compute_transition(state, action, reward, next_state, terminated, algorithm, agents, env)

            for agent, transition in zip(agents, transitions):
                agent.update_q(transition)

            state = next_state.copy()

            if env.render_mode:
                env.render()
        

        for agent, agent_eval in zip(agents, agents_eval):
            agent.update_epsilon()
            agent_eval.q = copy.deepcopy(agent.q)

        n_steps.append(env.steps)
        n_eps.append(agents[0].epsilon)

        n_steps_trained.append(run_trained(agents_eval, env_test, algorithm))
        lines = plot_graphs3(n_steps, n_eps, n_steps_trained, lines, iteractive=iteractive)

    # Save Informations after running
    with open(os.path.join('.', 'history', env.name + '.txt'), 'a') as file:
        for trajectory in env.trajectories:
            file.write(trajectory)

    with open(os.path.join('.', 'history', env_test.name + '.txt'), 'a') as file:
        for trajectory in env_test.trajectories:
            file.write(trajectory)
    plt.savefig(os.path.join('.', 'fig', env.name + '.png'))
