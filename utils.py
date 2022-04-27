import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch


sns.set(style='white')


def flatten_swarm_traj(a,agent_num,step):
    
    row=[]
    out=[]
    b=torch.chunk(a,agent_num,dim=0)
    
    for i in b:
        row.append(torch.chunk(i,step,dim=1))
       
    for i in range(agent_num):   
        out.append(torch.cat(row[i],dim=0))
    out=tuple(out)
    
    return torch.cat(out,dim=0)
    

    
  



def flatten_list_of_agents(list_agents):
    output=list_agents[0][0]
    for i in range(1,len(list_agents)):
        output=torch.cat((output,list_agents[i][0]))
    return output
    
# Flattens a list of dicts with torch Tensors


def flatten_list_dicts(list_dicts):
    
    out=[] 
    outt={}    
    keys1=['states', 'actions', 'rewards', 'terminals',  'old_log_prob_actions']
    keys2=['log_prob_actions','values','entropies']
    
    for k in keys1:
        
        for item in list_dicts:            
            out.append(item[k])
        
        outt[k]=torch.cat(tuple(out),dim=0)
       
        out=[]
   
    for k in keys2:
       
        for item in list_dicts:
            if (len(item[k].shape)==1):
                out.append(item[k].unsqueeze(1))
            else:
                out.append(item[k])
            outt[k]=torch.cat(tuple(out),dim=0)
            out=[]
    
    return outt


def gflatten_list_dicts(list_dicts):
    
    out=[] 
    outt={}    
    keys1=['states', 'actions',  'terminals',  'old_log_prob_actions']
    keys2=['log_prob_actions','values','rewards','entropies']
    
    for k in keys1:
        
        for item in list_dicts:            
            out.append(item[k])
        
        outt[k]=torch.cat(tuple(out),dim=0)
       
        out=[]
   
    for k in keys2:
       
        for item in list_dicts:
            if (len(item[k].shape)==1):
                out.append(item[k].unsqueeze(1))
            else:
                out.append(item[k])
            outt[k]=torch.cat(tuple(out),dim=0)
            out=[]
    
    return outt





#testtt=flatten_list_dicts(policy_trajectory_replay_buffer)



def flatten_list_dicts2(list_dicts):
    
  return {k: torch.cat([d[k] for d in list_dicts], dim=1) for k in list_dicts[-1].keys()}





# Makes a lineplot with scalar x and statistics of vector y
def lineplot(x, y, filename, xaxis='Steps', yaxis='Returns'):
  y = np.array(y)
  y_mean, y_std = y.mean(axis=1), y.std(axis=1)
  sns.lineplot(x, y_mean, color='coral')
  plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='coral', alpha=0.3)
  plt.xlim(left=0, right=x[-1])
  plt.ylim(bottom=0, top=500)  # Return limits for CartPole-v1
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  plt.savefig(os.path.join('results', filename + '.png'))
  plt.close()
