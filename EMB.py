# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:32:21 2021

@author: yx
"""

import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal
from ma_envs.envs.point_envs import rendezvous

class MeanEm(nn.Module):
    def __init__(self, nr_agents,state_size, hidden_size, local_size):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size,hidden_size),
            nn.ReLU(inplace=True)
        )
        self.state_size=state_size
        self.hidden_size=hidden_size
        self.local_size=local_size

        self.nr_agents=nr_agents
        
        self.em=nn.Linear(state_size,hidden_size)

    def forward(self,state):
        
        agents_state=state[:,:-self.local_size]
        
        agents_state=agents_state.reshape([-1,self.state_size])
        nei_num=agents_state.reshape(self.nr_agents,-1,self.state_size)[:,:,-2].sum(dim=1)
        
        one=torch.ones_like(nei_num)
        nei_num=torch.where(nei_num <one, one, nei_num)
        
        embed=self.feature_extractor(agents_state.reshape([-1,self.state_size]))

        embed=embed.reshape(self.nr_agents,-1,self.hidden_size).sum(dim=1)
        
        embed=torch.div(embed.T,nei_num).T
       
        return embed


class MLPActorCritic(nn.Module):

    def __init__(self, nr_agents, obs_dim, act_dim, hidden_size,local_size):
        super().__init__()
        self.feature = MeanEm(nr_agents,obs_dim, hidden_size,local_size)
        self.actor_fc= nn.Linear(hidden_size,act_dim)
        self.critic_fc= nn.Linear(hidden_size, 1)


       
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))



    def actor(self, state):
        x = torch.tanh(self.feature(state))
        mu = self.actor_fc(x)
        std = torch.exp(self.log_std)
        
        return Normal(mu, std)

    def critic(self, state):
        x = torch.tanh(self.feature(state))
        value = self.critic_fc(x)
        return torch.squeeze(value)

    def forward(self, state):
        pi = self.actor(state)
        v = self.critic(state)
        return pi, v

    def log_prob(self, state, action):
        return self.actor(state).log_prob(action).sum(axis=-1)