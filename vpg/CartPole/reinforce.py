from collections import deque
import torch

def reinforce(
        n_eps=1000, 
        max_t=1000, 
        gamma=1.0, 
        print_every=100,
        policy=None,
        optimizer=None
    ):
    scores_deque = deque(maxlen=100)