import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import Perception_Module


class Actor(nn.Module):
    """Actor network (policy network) for SAC."""

    def __init__(self, num_actions, goal_dim=0, depth_only=False, pooled_size=8):
        super(Actor, self).__init__()
        self.perception = Perception_Module()
        self.spatial_pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        feature_size = 512 * pooled_size * pooled_size
        self.goal_dim = goal_dim
        self.fc1 = nn.Linear(feature_size + goal_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        self.depth_only = depth_only

    def forward(self, state, goal=None):
        """
        Args:
            state: (batch_size, 4, 200, 200) or (batch_size, 1, 200, 200) for depth_only
        Returns:
            logits: (batch_size, num_actions) - unnormalized log probabilities
        """
        x = self.perception(state)
        x = self.spatial_pool(x)
        # Flatten the feature map
        x = x.view(x.size(0), -1)
        if self.goal_dim and goal is not None:
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)
            x = torch.cat([x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_and_log_prob(self, state, goal=None):
        """
        Sample an action from the policy and compute its log probability.
        """
        logits = self.forward(state, goal)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class QNetwork(nn.Module):
    """Critic estimating Q(s,a) pairs."""

    def __init__(self, num_actions, goal_dim=0, depth_only=False, pooled_size=8):
        super(QNetwork, self).__init__()
        self.perception = Perception_Module()
        self.spatial_pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        feature_size = 512 * pooled_size * pooled_size
        self.goal_dim = goal_dim
        # Embedding for action (one-hot encoding)
        self.action_embedding = nn.Embedding(num_actions, 128)
        # Combine state features with action embedding
        self.fc1 = nn.Linear(feature_size + 128 + goal_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.depth_only = depth_only

    def forward(self, state, action, goal=None):
        """
        Args:
            state: (batch_size, 4, 200, 200) or (batch_size, 1, 200, 200)
            action: (batch_size,) - action indices
        Returns:
            q_value: (batch_size, 1) - Q-value for state-action pair
        """
        # Extract state features
        state_features = self.perception(state)
        state_features = self.spatial_pool(state_features)
        state_features = state_features.view(state_features.size(0), -1)

        # Embed action
        action_emb = self.action_embedding(action)

        # Concatenate state features and action embedding
        features = [state_features, action_emb]
        if self.goal_dim and goal is not None:
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)
            features.append(goal)
        x = torch.cat(features, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
