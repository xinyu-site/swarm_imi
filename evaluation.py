import torch

import numpy as np
from ma_envs.envs.point_envs import rendezvous
from utils import flatten_list_dicts2, flatten_swarm_traj,flatten_list_dicts


'''
# Evaluate agent with deterministic policy π
def evaluate_agent(agent, episodes, return_trajectories=False, seed=1):
    env = rendezvous.RendezvousEnv(nr_agents=10,
                                       obs_mode='fix_acc',
                                       comm_radius=100 * np.sqrt(2),
                                       world_size=100,
                                       distance_bins=8,
                                       bearing_bins=8,
                                       torus=False,
                                       dynamics='unicycle_acc')

    returns, trajectories = [], []
    for _ in range(episodes):
        states, actions, rewards = [], [], []
        state, terminal = env.reset(), False
        while not terminal:
            with torch.no_grad():
                policy, _ = agent(state)
                action = policy.sample()  # Pick action greedily
                state, reward, terminal,info = env.step(action.numpy())
    
            if return_trajectories:
                states.append(state)
                actions.append(action)
            rewards.append(reward)
        returns.append(sum(rewards))
    
    if return_trajectories:
        
        
        
        
      # Collect trajectory data (including terminal signal, which may be needed for offline learning)
      terminals = torch.cat([torch.ones(len(rewards) - 1), torch.zeros(1)])
      
      
      
      trajectories.append(dict(states=torch.cat(states), actions=torch.cat(actions), rewards=torch.tensor(rewards, dtype=torch.float32), terminals=terminals))

    return (returns, trajectories) if return_trajectories else returns
'''


def evaluate_agent(agent, episodes):
    
    env = rendezvous.RendezvousEnv(nr_agents=10,
                                           obs_mode='fix_acc',
                                           comm_radius=100 * np.sqrt(2),
                                           world_size=100,
                                           distance_bins=8,
                                           bearing_bins=8,
                                           torus=False,
                                           dynamics='unicycle_acc')
    
    
    episodes=10
    trajectories = []
    
    state, terminal = env.reset(), False
    for i in range(episodes*env.timestep_limit):
        with torch.no_grad():
            policy,value = agent(state)
            action = policy.sample()  # Pick action greedily
            next_state, reward, terminal,info = env.step(action.numpy())
            
            trajectories.append(dict(states=state,actions=action,rewards=reward.unsqueeze(1),terminals = torch.tensor([terminal]*env.nr_agents, dtype=torch.float32).unsqueeze(1)))
            state=next_state
            
            if terminal:
                state,terminal=env.reset(),False
         
    trajectories=flatten_list_dicts2(trajectories)
    trajectories={k: flatten_swarm_traj(trajectories[k],env.nr_agents,episodes*env.timestep_limit) for k in trajectories.keys()}
    return trajectories


