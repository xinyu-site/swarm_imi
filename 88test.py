# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:33:19 2021

@author: yx
"""

from pylab import *
import numpy as np
from scipy.interpolate import griddata
import torch

from ma_envs.envs.point_envs import rendezvous
import seaborn as sns
from ppo_model import MLPActorCritic,AIRLDiscriminator
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings("ignore")

font=13
seed=0

nr_agents=10
np.random.seed(seed)

torch.manual_seed(0)
env = rendezvous.RendezvousEnv(nr_agents=10,obs_mode='fix_acc',comm_radius=100 * np.sqrt(2),world_size=100,distance_bins=8,bearing_bins=8,torus=False,dynamics='unicycle')




agent = MLPActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 128,True)
discriminator = AIRLDiscriminator(env.observation_space.shape[0], env.action_space.shape[0], 128, 0.99, state_only=True)
agent.load_state_dict(torch.load('D:/asu/unicycle531/531unicycleagent.pth'))
discriminator.load_state_dict(torch.load('D:/asu/unicycle531/531unicyclediscriminator.pth'))



nr_agents=10
time_step=110

agentid=8


np.random.seed(seed)
st0=env.reset()
for t in range(time_step):    
    st0=env.step(agent(st0)[0].sample().numpy())[0]
    
    
state_t=env.world.agent_states
state0=env.set_states(state_t)


jiancha=[]
action0=np.zeros((10,2))
a = np.arange(-1, 1, 0.1)
b = np.arange(0,1,0.05)
midu=20



a0,a1=np.meshgrid(b,a)
action_set=[]



for i in range(len(a)):
    for j in range(len(a)):
        action_set.append(np.array((a0[i][j],a1[i][j])))
        


        
action_set=np.array(action_set).reshape(midu,midu,2)
    
heat_data=np.random.rand(midu,midu)

for i in range(midu):
    for j in range(midu):
        np.random.seed(0)
        state0=env.set_states(state_t)
        action0[agentid]=action_set[i][j]
        jiancha.append(copy.deepcopy(action0))
        state1=env.step(action0)[0]
        lop=agent.log_prob(state0,torch.tensor(action0)).exp()[agentid]
        heat=discriminator.predict_reward(state0,action0,state1,lop,False)[agentid]
        
        #heat=discriminator.g(state1)[agentid]
        heat_data[i][j]=heat.item()










#create 5000 Random points distributed within the circle radius 100
max_r = 100
max_theta = 2.0 * np.pi
number_points = 5000
points = np.random.rand(number_points,2)*[max_r,max_theta]

#Some function to generate values for these points, 
#this could be values = np.random.rand(number_points)
values = points[:,0] * np.sin(points[:,1])* np.cos(points[:,1])

#now we create a grid of values, interpolated from our random sample above
theta = np.arange(-1,1, 0.1)
r = np.linspace(0, max_r, 20)
grid_r, grid_theta = np.meshgrid(r, theta)



data = griddata(points, values, (grid_r, grid_theta), method='cubic',fill_value=0)


for i in range(20):
    data[i,:]=heat_data[-1][i]
    print(data[i])


#Create a polar projection
ax1 = plt.subplot(projection="polar")
ax1.pcolormesh(theta,r,heat_data)
plt.show()