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

        # for N leafs, there has to be N-1 nodes above, so total adds up to 2N-1
        self.tree = np.zeros(2 * capacity - 1)
        # The tree only stores N leafs, as branch nodes are used for O(Log N) search
        self.data = np.zeros(capacity, dtype=object)

        self.ptr = 0
        self.n_entries = 0
        
        # Small constant to ensure no transition has 0 prob
        self.epsilon = 0.01 

    def __len__(self):
        return self.n_entries


    def add(self, data):

        # Find current max priority in the leaves to ensure new data gets sampled at least once
        max_p = np.max(self.tree[-self.capacity:])
        if max_p <= 0:
            max_p = 1.0 # Initial priority for the very first entry

        # Calculate the leaf index in the 'tree' array
        # The data at self.data[0] is mapped to tree[capacity - 1]
        tree_idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data


        # Update tree with this priority
        self.update(tree_idx, max_p) 

        
        # Move pointer (circular buffer logic)
        # this has the same effect as the previously used Deque
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)


    def update(self, idx, priority):
        # 1. Calculate new priority with alpha-scaling: p = (|error| + e)^alpha
        p = (abs(priority) + self.epsilon) ** self.alpha
        
        
        # 2. Find the difference between new priority and current value
        change = p - self.tree[idx]
        self.tree[idx] = p
        
        # 3. Propagate the change up to the root (index 0)
        # This is O(log N). Each parent is the sum of its two children.
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change


    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we've reached the bottom of the tree, this is our leaf
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Binary search: if v is less than left child, go left.
            # Otherwise, subtract left child value from v and go right.
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        # Calculate index for 'data' array based on tree leaf index
        data_idx = leaf_idx - self.capacity + 1
        # (idx, Priority, Sample data)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def sample(self, n):
        batch = []
        idxs = []
        priorities = []
        
        # Divide the total sum (root of tree) into 'n' equal segments
        segment = self.tree[0] / n
        
        # Move beta towards 1.0 to reduce bias as training progresses
        self.beta = min(1., self.beta + self.beta_increment)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            # Pick a random value from each segment
            s = random.uniform(a, b)
            idx, p, data = self.get_leaf(s)
            
            priorities.append(p)
            idxs.append(idx)
            batch.append(data)

        # Importance Sampling Weights: (1/N * 1/P(i))^beta
        # This corrects for the fact that we are sampling non-uniformly.
        sampling_probabilities = np.array(priorities) / self.tree[0]
        is_weights = np.power(self.n_entries * sampling_probabilities, -self.beta)
        
        # Normalize weights so they only scale gradients down (max weight = 1.0)
        is_weights /= is_weights.max() 

        return batch, idxs, torch.FloatTensor(is_weights)