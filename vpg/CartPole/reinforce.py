from collections import deque
import torch
import numpy as np

def reinforce(
        n_eps=1000, 
        max_t=1000, 
        gamma=1.0, 
        print_every=100,
        env=None,
        policy=None,
        optimizer=None
    ):
    scores_deque = deque(maxlen=100)
    scores = []
    for i in range(1, n_eps+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i % print_every == 0:
            print(f"Episode: {i}, average score: {np.mean(scores_deque)}.")
        if np.mean(scores_deque) >= 195.0:
            print(f"Cartpole env solved in {i} episodes! Average score: {np.mean(scores_deque)}.")
            break
    return scores
        