import numpy as np
import torch
from policy import Policy
import time
#import gym
import gymnasium as gym
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = Policy(device=device)
policy.load_state_dict(torch.load("policy_cartpole.pth"))

env = gym.make('CartPole-v1')
state = env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
for t in range(1000):
    action, _ = policy.act(state)
    env.render()
    time.sleep(0.1)

    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()


