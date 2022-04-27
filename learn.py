# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 00:13:23 2021

@author: yx
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:38:46 2020

@author: yx
"""

from collections import deque
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from ma_envs.envs.point_envs import rendezvous
from ppo_model import MLPActorCritic, AIRLDiscriminator
from utils import flatten_list_dicts2, flatten_swarm_traj,flatten_list_dicts

import matplotlib.pyplot as plt
import seaborn as sns


from training import TransitionDataset, adversarial_imitation_update,  compute_advantages, ppo_update, target_estimation_update


def evaluate_agent_rez(agent, episodes,env):
    reward_list=[]
    #rew=[]
    t=0
    
    state, terminal = env.reset(), False
    for i in range(episodes*env.timestep_limit):
        with torch.no_grad():

            
 
            
            
            
            action=agent(state)[0].sample()
            next_state, reward, terminal,info = env.step(action.numpy())
            
            reward_list.append(reward.mean().item())
            env.render(mode='human')
          
            state=next_state
            t=t+1
            if terminal:
                state,terminal=env.reset(),False
            
    return 0
    #return reward_list


def airl_learn(exp_logdir,n_agents,exp_len,seed,steps,hidden_size,discount,trace_decay,ppo_clip,ppo_epochs,value_loss_coeff,entropy_loss_coeff,learning_rate,batch_size,imitation,state_only,\
          imitation_epochs,imitation_batch_size,imitation_replay_size,r1_reg_coeff):
    
   
    torch.manual_seed(seed)
    
    env = rendezvous.RendezvousEnv(nr_agents=n_agents,
                                   obs_mode='fix_acc',
                                   comm_radius=100,
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle')
    
    agent = MLPActorCritic(env.observation_space.shape[0], 2, hidden_size,True)
    
    agent_optimiser = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    
    
    
    env.seed(seed)
    # Set up expert trajectories dataset
    #expert_trajectories = TransitionDataset(torch.load('results/vicsek_data.pth'))
    expert_trajectories = TransitionDataset(torch.load(exp_logdir))      
    expert_episodes=int((expert_trajectories.__len__()+1)/512/env.nr_agents)
        
    discriminator = AIRLDiscriminator(env.observation_space.shape[0], 1, hidden_size, discount, state_only=state_only)
     
    discriminator_optimiser = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)
    # Metrics
    metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])
    
    # Main training loop
    state, terminal, episode_return, trajectories, policy_trajectory_replay_buffer = env.reset()\
    , False, 0, [], deque(maxlen=imitation_replay_size)
    
    
    pbar = tqdm(range(1, steps + 1), unit_scale=1, smoothing=0)
    for step in pbar:
        
       
        # Collect set of trajectories by running policy π in the environment
    
        if  imitation != 'BC':

            
            
            
            policy, value = agent(state)
            action = policy.sample()
            log_prob_action, entropy = agent.log_prob(state,action), policy.entropy().sum(axis=-1)
            next_state, reward, terminal ,info= env.step(action.numpy())
    
            episode_return += reward
            trajectories.append(dict(states=state,
                                     actions=action,
                                     rewards=reward.unsqueeze(1),
                                     terminals = torch.tensor([terminal]*env.nr_agents, dtype=torch.float32).unsqueeze(1),
                                     log_prob_actions=log_prob_action.unsqueeze(1), 
                                     old_log_prob_actions=log_prob_action.detach().unsqueeze(1),
                                     values=value.unsqueeze(1), 
                                     entropies=entropy.unsqueeze(1)))
            state = next_state
            
            
            
            if terminal:
                    
              # Store metrics and reset environment
                metrics['train_steps'].append(step)
                metrics['train_returns'].append([episode_return])
                pbar.set_description('Step: %i | Return: %f' % (step, episode_return.mean()))
               
                state, episode_return = env.reset(), 0
                                
                trajectories=flatten_list_dicts2(trajectories)                
                trajectories={k: flatten_swarm_traj(trajectories[k],env.nr_agents,env.timestep_limit) for k in trajectories.keys()}
         
                if trajectories['terminals'].shape[0] >= batch_size:              
         
                    policy_trajectories=trajectories 
                    trajectories = []  # Clear the set of trajectories          
                    policy_trajectory_replay_buffer.append(policy_trajectories)
                    policy_trajectory_replays = flatten_list_dicts(policy_trajectory_replay_buffer)
                    
                    for _ in tqdm(range(imitation_epochs), leave=False):
                        
                        adversarial_imitation_update(imitation, agent, discriminator, expert_trajectories, TransitionDataset(policy_trajectory_replays), discriminator_optimiser, imitation_batch_size, r1_reg_coeff)
                         
                    # Predict rewards
                    with torch.no_grad():
                        policy_trajectories['rewards'] = discriminator.predict_reward(policy_trajectories['states'], policy_trajectories['actions'], torch.cat([policy_trajectories['states'][1:], next_state[-1].unsqueeze(0)]), policy_trajectories['log_prob_actions'].exp(), policy_trajectories['terminals'])
                          
                        
                    # Compute rewards-to-go R and generalised advantage estimates ψ based on the current value function V
                    compute_advantages(policy_trajectories, agent(state)[1][0], discount, trace_decay)
                    # Normalise advantages
                    policy_trajectories['advantages'] = (policy_trajectories['advantages'] - policy_trajectories['advantages'].mean()) / (policy_trajectories['advantages'].std() + 1e-8)
                   #torch.save(policy_trajectories, os.path.join('DEBUG', 'trajectories.pth'))
                    
                    # Perform PPO updates
                    for epoch in tqdm(range(ppo_epochs), leave=False):
                      
                      ppo_update(agent, policy_trajectories, agent_optimiser, ppo_clip, epoch, value_loss_coeff, entropy_loss_coeff)
                
                
               
    torch.save(agent.state_dict(), os.path.join('airl_res','test', 'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, expert_episodes, env.obs_mode)+'agent.pth'))
    
    torch.save(discriminator.state_dict(), os.path.join('airl_res','test', 'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, expert_episodes, env.obs_mode)+'discriminator_airl.pth'))
    torch.save(metrics, os.path.join('airl_res','test', 'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, expert_episodes, env.obs_mode)+'metrics.pth'))
    
    rew=[]
    time_of_repeat=10
    for i in range(time_of_repeat):
        rew.append(evaluate_agent_rez(agent,1,env))
        
    torch.save(rew, os.path.join('airl_res','test', 'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, expert_episodes, env.obs_mode)+'rew.pth'))
    
  
    plt.figure()
    colorlist = ['r', 'g', 'b', 'k']
    def smooth(data, sm=1):
        if (sm > 1):
            smooth_data = []
            for d in data:
                y = np.ones(sm)*1.0/sm
                d = np.convolve(y, d, "same")
    
                smooth_data.append(d)

        return smooth_data
    x=range(512)
    
    sns.set(style="darkgrid", font_scale=1)
    
    sns.tsplot(time=x, data=smooth(rew,sm=2), color=colorlist[1], condition="airl")

    plt.legend(loc='1')
    
    plt.ylabel("order_parameter")
    plt.xlabel("Iteration Number")
    plt.title("vicsek")
    
    
    
    plt.savefig(os.path.join('airl_res','test', 'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, expert_episodes, env.obs_mode)+'ren_pic.png'))
    
    
    env.close()      



