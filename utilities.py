import matplotlib.pyplot as plt
import numpy as np
import copy

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

def plot_graphs3(n_steps, n_eps, n_steps_trained, lines):
    lines[0].set_data(np.arange(len(n_steps)), n_steps)
    lines[1].set_data(np.arange(len(n_steps_trained)), n_steps_trained)
    lines[2].set_data(np.arange(len(n_eps)), n_eps)

    for line in lines:
        line.axes.relim()
        line.axes.autoscale_view()

    plt.draw()
    plt.pause(0.01)

    return lines

def convert_old(state, env_length, env):
    fly_pos = env.fly_dict[str(state[0:2])]
    if fly_pos == 3:
        return 3 * env_length
    spider_pos = state[2]
    return int(fly_pos * env_length + spider_pos)

def convert(state, env_length, env):
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
    # agent.epsilon = 0
    for _ in range(epochs):

        state, _ = env_test.reset()
        state = convert(state, agent[0].state_len, env_test)
        if env_test.render_mode:
            env_test.render()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            
            # action1 = agent.select_action(state)
            action = compute_action(state, agent, algorithm)

            # action = (action1,)
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            next_state = convert(next_state, agent[0].state_len, env_test)

            state = next_state

            if env_test.render_mode:
                env_test.render()

        #n_steps.append(env.steps)

    return env_test.steps

def compute_action(state, agents, algorithm):
    # return [agent.select_action(state) for agent in agents]
    if algorithm == "q-learning":
        state = convert(state, agents[0].state_len, env)
        return (agents[0].select_action(state), )
    
    if algorithm == "iql":
        return [agent.select_action(state) for agent in agents]


def compute_transition(state, action, reward, next_state, terminated, algorithm):
    if algorithm == "q-learning":
        transitions = state, int(action[0] + 1), reward, next_state, terminated
        return (transitions,)

    if algorithm == "iql":
        transition1 = state, int(action[0] + 1), reward, next_state, terminated
        transition2 = state, int(action[1] + 1), reward, next_state, terminated
        return (transition1, transition2)

def train(epochs, env, agents, algorithm):
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

    # agent1 = agents[0]
    agents_eval = [Agent(agent.state_len, agent.action_len, agent.eps_decay, eps_min=0., eps_max=0.) for agent in agents]

    for _ in range(epochs):
        # cum_rew = 0

        state, _ = env.reset()
        # state = convert(state, agents[0].state_len, env)

        # print(state)
        if env.render_mode:
            env.render()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            
            action = compute_action(state, agents, algorithm)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = convert(next_state, agents[0].state_len, env)

            transitions = compute_transition(state, action, reward, next_state, terminated, algorithm)

            for agent, transition in zip(agents, transitions):
                agent.update_q(transition)


            state = next_state
            #cum_rew += reward
            # print(next_state)
            if env.render_mode:
                env.render()
        

        for agent, agent_eval in zip(agents, agents_eval):
            agent.update_epsilon()
            agent_eval.q = copy.deepcopy(agent.q)

        n_steps.append(env.steps)
        n_eps.append(agents[0].epsilon)

        # agent_eval.q = copy.deepcopy(agent1.q)
        n_steps_trained.append(run_trained(agents_eval, env_test, algorithm))
        lines = plot_graphs3(n_steps, n_eps, n_steps_trained, lines)
