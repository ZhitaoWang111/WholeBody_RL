import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP_Agent(nn.Module):
    def __init__(self, action_space, sample_obs, Encoder, device):
        super().__init__()
        self.feature_net = Encoder
        latent_size = self.feature_net.feature_size
        self.n_actions = action_space.shape[0]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, self.n_actions), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, self.n_actions) * -0.5)

    def get_features(self, x):
        return self.feature_net(x)
    
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x.detach())
    
    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # clamp for numerical stability
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return torch.tanh(action_mean)
        action_logstd = self.actor_logstd.clamp(-20, 2).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        raw_action = dist.sample()
        return torch.tanh(raw_action)
    
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.clamp(-20, 2).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        if action is None:
            raw_action = dist.sample()
            action = torch.tanh(raw_action)
            squashed_action = action
        else:
            squashed_action = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
            raw_action = self._atanh(squashed_action)
        # log prob with tanh correction
        logprob = dist.log_prob(raw_action).sum(1)
        logprob -= torch.log(1 - squashed_action.pow(2) + 1e-6).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(x.detach())
