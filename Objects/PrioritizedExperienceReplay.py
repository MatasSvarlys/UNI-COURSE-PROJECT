import random

import numpy as np
import torch


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        # How much prioritization to use (0 = uniform, 1 = full)
        self.alpha = alpha  
        # Importance sampling weight
        self.beta = beta    
        self.beta_increment = beta_increment
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.n_entries = 0
        
        # Small constant to ensure no transition has 0 prob
        self.epsilon = 0.01 

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        p = priority ** self.alpha
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        return leaf_idx, self.tree[leaf_idx], self.data[leaf_idx - self.capacity + 1]

    def sample(self, n):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree[0] / n
        self.beta = min(1., self.beta + self.beta_increment)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.get_leaf(s)
            priorities.append(p)
            idxs.append(idx)
            batch.append(data)

        # Importance Sampling Weights: (1/N * 1/P(i))^beta
        sampling_probabilities = np.array(priorities) / self.tree[0]
        is_weights = np.power(self.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() # Normalize

        return batch, idxs, torch.FloatTensor(is_weights)