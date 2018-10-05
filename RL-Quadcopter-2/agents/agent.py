import numpy as np
from utils.ddpg import DDPG
from task import Task

class DDPG_Agent():
    def __init__(self, task):
        self.ddpg = DDPG(task)

    def reset_episode(self):
        return self.ddpg.reset_episode()

    def step(self, action, reward, next_state, done):
        self.ddpg.step(action, reward, next_state, done)
    
    def act(self, state, enable_exploration = True):
        return self.ddpg.act(state, enable_exploration)
    
    def load_model(self, actor_filename, critic_filename):
        self.ddpg.load_model(actor_filename, critic_filename)
        
    def save_model(self, actor_filename, critic_filename):
        self.ddpg.save_model(actor_filename, critic_filename)