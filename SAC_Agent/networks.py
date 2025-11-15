import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import Perception_Module


class Actor(nn.Module):
    """
    Actor network (policy network) for SAC.
    Outputs a categorical distribution over discrete actions.
    """

    def __init__(self, num_actions, depth_only=False):
        super(Actor, self).__init__()
        self.perception = Perception_Module()
        # Perception module outputs (batch, 512, 50, 50) after pooling
        # 200x200 -> MP1 (stride 2) -> 100x100 -> MP2 (stride 2) -> 50x50
        feature_size = 512 * 50 * 50
        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        self.depth_only = depth_only

    def forward(self, state):
        """
        Args:
            state: (batch_size, 4, 200, 200) or (batch_size, 1, 200, 200) for depth_only
        Returns:
            logits: (batch_size, num_actions) - unnormalized log probabilities
        """
        x = self.perception(state)
        # Flatten the feature map
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_and_log_prob(self, state):
        """
        Sample an action from the policy and compute its log probability.
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class QNetwork(nn.Module):
    """
    Q-network (critic) for SAC.
    Estimates Q(s,a) for state-action pairs.
    """

    def __init__(self, num_actions, depth_only=False):
        super(QNetwork, self).__init__()
        self.perception = Perception_Module()
        # Perception module outputs (batch, 512, 50, 50)
        feature_size = 512 * 50 * 50
        # Embedding for action (one-hot encoding)
        self.action_embedding = nn.Embedding(num_actions, 128)
        # Combine state features with action embedding
        self.fc1 = nn.Linear(feature_size + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.depth_only = depth_only

    def forward(self, state, action):
        """
        Args:
            state: (batch_size, 4, 200, 200) or (batch_size, 1, 200, 200)
            action: (batch_size,) - action indices
        Returns:
            q_value: (batch_size, 1) - Q-value for state-action pair
        """
        # Extract state features
        state_features = self.perception(state)
        state_features = state_features.view(state_features.size(0), -1)

        # Embed action
        action_emb = self.action_embedding(action)

        # Concatenate state features and action embedding
        x = torch.cat([state_features, action_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
