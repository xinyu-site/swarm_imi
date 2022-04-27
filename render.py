# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:37:13 2020

@author: yx
"""
import torch
import numpy as np
from ma_envs.envs.point_envs import rendezvous
from ppo_model import MLPActorCritic

import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


env = rendezvous.RendezvousEnv(nr_agents=10,
                                   obs_mode='fix_acc',
                                   comm_radius=100 * np.sqrt(2),
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle')



hidden_size=64

agent=MLPActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size,True)



'''
# Evaluate visulation.
render_eval = True
render_mode = 'human'
nb_eval_steps = 1024
episodes = 1

for ep in range(episodes):
    ob = env.reset()
    if render_eval:
        env.render(mode=render_mode)

    for t_rollout in range(nb_eval_steps):
        ac = agent(ob)[0].sample().numpy()
        ob, r, done, info = env.step(ac)
        if render_eval:
            env.render(mode=render_mode)
        if done or t_rollout == nb_eval_steps - 1:
            obs = env.reset()
            break
'''



'''
_, trajectories = evaluate_agent(agent, 500, return_trajectories=True, seed=0)
torch.save(trajectories, os.path.join('results', 'trajectories.pth'))    
'''


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
            env.render(mode='human')
            if terminal:
                state,terminal=env.reset(),False
         
    trajectories=flatten_list_dicts2(trajectories)
    trajectories={k: flatten_swarm_traj(trajectories[k],env.nr_agents,episodes*env.timestep_limit) for k in trajectories.keys()}
    return trajectories
'''




def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return smooth_data

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
            
            '''
            lop=agent.log_prob(state,action).exp()
            lop=lop.reshape((10,1))
            print(state.shape)
            print(action.shape)
            print(next_state.shape)
            print(lop.shape)
            
            #heat=discriminator.predict_reward(state,action,next_state,lop,False)
            
            #rew.append(dis.g(next_state))
            rew.append(dis.predict_reward(state,action,next_state,lop,False))
           
            if t % 1 == 0:
                env.render(mode='human')
            '''
            state=next_state
            t=t+1
            if terminal:
                state,terminal=env.reset(),False
            
   
    return reward_list

def evaluate_agent_random(episodes,env):
    reward_list=[]
    
    t=0
    
    state, terminal = env.reset(), False
    for i in range(episodes*env.timestep_limit):
        with torch.no_grad():

            
            a = 2 * np.random.rand(env.nr_agents, 2) - 1
            
            
            
            
            next_state, reward, terminal,info = env.step(a)
            
            reward_list.append(reward.mean().item())

            '''
            if t % 1 == 0:
                env.render(mode='human')
            '''
            state=next_state
            t=t+1
            if terminal:
                state,terminal=env.reset(),False
            
   
    return reward_list


time_of_repeat=10

reward_exp=[]

reward_airl=[]
reward_rand=[]
reward_exp2=[]
reward_bc=[]

    
'''
agent.load_state_dict(torch.load("results/agent.pth"))
for i in range(time_of_repeat):
    reward_airl.append(evaluate_agent_rez(agent,1,env))  
'''


agent.load_state_dict(torch.load('D:/asu/unicycle531/bc_agent_red.pth'))

for i in range(time_of_repeat):
    reward_bc.append(evaluate_agent_rez(agent,1,env))
 
 
for i in range(time_of_repeat):
    reward_rand.append(evaluate_agent_random(1,env))    

agent = MLPActorCritic(50, 2, 128,True)
agent.load_state_dict(torch.load('D:/asu/unicycle531/expert_agent.pth'))
for i in range(time_of_repeat):
    reward_exp.append(evaluate_agent_rez(agent,1,env))
    
'''   
agent = MLPActorCritic(50, 2, 128,True)
agent.load_state_dict(torch.load('D:/asu/530agent.pth'))
for i in range(time_of_repeat):
    reward_exp2.append(evaluate_agent_rez(agent,1,env))    
'''

agent = MLPActorCritic(50, 2, 128,True)
agent.load_state_dict(torch.load('D:/asu/unicycle531/531unicycleagent.pth'))
for i in range(time_of_repeat):
    reward_airl.append(evaluate_agent_rez(agent,1,env))    







plt.figure()
colorlist = ['r', 'g', 'b', 'k']
x=range(512)
sns.set(style="darkgrid", font_scale=1)

'''
sns.tsplot(time=x, data=smooth(reward_exp,sm=2),color=colorlist[0], condition="Exp")
sns.tsplot(time=x, data=smooth(reward_bc,sm=2), color=colorlist[1], condition="Airl")
sns.tsplot(time=x, data=smooth(reward_rand,sm=2), color=colorlist[2], condition="Random")

sns.tsplot(time=x, data=smooth(reward_airl,sm=2), color=colorlist[3], condition="Bc")
'''


sns.tsplot(time=x, data=reward_exp,color=colorlist[0], condition="Expert")
sns.tsplot(time=x, data=reward_bc, color=colorlist[1], condition="Ps-Airl")
sns.tsplot(time=x, data=reward_rand, color=colorlist[2], condition="Random")

sns.tsplot(time=x, data=reward_airl,color=colorlist[3], condition="BC")




plt.legend(loc='1')

plt.ylabel("reward")
plt.xlabel("Iteration Number")
plt.title("rendezvous")


plt.show()


