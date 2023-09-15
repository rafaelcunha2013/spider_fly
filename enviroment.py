import numpy as np
import copy
import os

class SpiderFly2D:

    def __init__(self, spiders, flies, max_steps=50, render_mode=False, size=12,
                 name='test.txt', print_li = False):
        self.spiders = spiders
        self.flies = flies
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.size = size
        self.name = name
        self.print_li = print_li
        
        self.grid = np.array(['___']*size, dtype='U3')
        self.steps = 0
        
        self.state = np.empty((len(spiders) + len(flies),), dtype=int)

        self.fly_dict = {
                str(np.array([flies[0].position, flies[1].position])) : 0,
                str(np.array([-1, flies[1].position])): 1,
                str(np.array([flies[0].position, -1])) : 2,
                str(np.array([-1, -1])): 3
        }
        self.trajectories = []

    def update_grid(self):
        self.grid = np.array(['___']*self.size, dtype='U3')

        for fly in self.flies:
            self.grid[fly.position] = fly.name

        for spider in self.spiders:
            self.grid[spider.position] = spider.name

    def update_state(self):
        i = 0
        for fly in self.flies:
            self.state[i] = fly.position if not fly.captured else -1
            i += 1

        for spider in self.spiders:
            self.state[i] = spider.position
            i += 1


    def reset(self):
        [fly.reset() for fly in self.flies]
        [spider.reset() for spider in self.spiders]
        self.update_state()
        if self.render_mode:
            self.update_grid()
        self.steps = 0

        self.trajectories.append("\n\n")

        return self.state, {}
    
    def step(self, action):
        reward = 0
        for spider, ind_action in zip(self.spiders, action):
            old_position = copy.deepcopy(spider.position)
            spider.move(ind_action)
            if spider.position == len(self.grid) or spider.position == -1:
                spider.position = copy.deepcopy(old_position)

        for fly in self.flies:
            for spider in self.spiders:
                if fly.position == spider.position and fly.captured == False:
                    fly.captured = True
                    fly.name = '___'
                    reward += 16
        self.update_state()
        if self.render_mode:
            self.update_grid()

        terminated = True if sum([fly.captured for fly in self.flies]) == len(self.flies) else False
        self.steps += 1
        truncated = True if (self.steps == self.max_steps and not terminated) else False


        return self.state, reward, terminated, truncated, {}

    def render(self):
        self.trajectories.append('   '.join(self.grid) + "\n")
        if self.print_li:
            print('   '.join(self.grid))


    
class Spider:

    def __init__(self, position, name):
        self.init_position = position
        self.name = name

        self.action = None
        self.position = self.init_position

    def move(self, action):
        self.position += action

    def choose_action(self):
        return np.random.choice([-1, 0, 1], size=1)
    
    def reset(self):
        self.position = self.init_position


class Fly:

    def __init__(self, position, name):
        self.init_position = position
        self.init_name = name

        self.captured = False
        self.position = self.init_position
        self.name = self.init_name

    def move(self, action):
        self.position += action

    def action(self):
        return np.random.choice([-1, 0, 1], size=1)
    
    def reset(self):
        self.postion = self.init_position
        self.captured = False
        self.name = self.init_name
    

if __name__ == "__main__":
    fly1 = Fly(0, 'F1 ')
    fly2 = Fly(11, 'F2 ')
    spd1 = Spider(5, 'Sp1')
    spd2 = Spider(6, 'Sp2')

    spiders = (spd1, spd2)
    flies = (fly1, fly2)

    render = True
    env = SpiderFly2D(spiders, flies, render_mode=render)
    state, _ = env.reset()
    # print(state)
    if render:
        env.render()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = tuple([spider.choose_action() for spider in spiders])
        next_state, reward, terminated, truncated, _ = env.step(action)
        # print(next_state)
        if render:
            env.render()

    print(env.steps)


        

