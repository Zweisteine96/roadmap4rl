r"""Train the algorithm of REINFORCE with Openai Gym's Cartpole env."""
#import gym
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import logging
from policy import Policy
from reinforce import reinforce

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')
#env.seed(0)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.info(f"env: {env}")
logging.info(f"observation space: {env.observation_space}")
logging.info(f"action space: {env.action_space}")

policy = Policy(device=device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
scores = reinforce(env=env, policy=policy, optimizer=optimizer)
torch.save(policy.state_dict(), "policy_cartpole.pth")

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()