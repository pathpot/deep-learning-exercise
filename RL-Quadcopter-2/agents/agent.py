import numpy as np
from task import Task

class ActorCritic_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

    def reset_episode(self):
        state = self.task.reset()
        return state

    
    def step(self, reward, done):
        pass
    
    def act(self, state):
        action = np.array([-66.89574554, 287.4330843, -220.56947131, 347.70047904])
        return action

    def learn(self):
        pass