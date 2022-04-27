# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 00:12:58 2021

@author: yx
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:41:48 2021

@author: yx
"""
import argparse

import itertools
import os
import learn


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='IL')

parser.add_argument('--n_agents', type=int, default=10)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--steps', type=int, default=150000, metavar='T', help='Number of environment steps')
parser.add_argument('--hidden-size', type=int, default=64, metavar='H', help='Hidden size')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount')
parser.add_argument('--trace-decay', type=float, default=0.95, metavar='λ', help='GAE trace decay')
parser.add_argument('--ppo-clip', type=float, default=0.2, metavar='ε', help='PPO clip ratio')
parser.add_argument('--ppo-epochs', type=int, default=4, metavar='K', help='PPO epochs')
parser.add_argument('--value-loss-coeff', type=float, default=1, metavar='c1', help='Value loss coefficient')
parser.add_argument('--entropy-loss-coeff', type=float, default=0, metavar='c2', help='Entropy regularisation coefficient')
parser.add_argument('--learning-rate', type=float, default=2e-4, metavar='η', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=2048, metavar='B', help='Minibatch size')
parser.add_argument('--imitation', type=str, default='AIRL', choices=['AIRL', 'BC', 'GAIL', 'GMMIL'], metavar='I', help='Imitation learning algorithm')
parser.add_argument('--state-only', action='store_true', default=True, help='State-only imitation learning')
parser.add_argument('--imitation-epochs', type=int, default=5, metavar='IE', help='Imitation learning epochs')
parser.add_argument('--imitation-batch-size', type=int, default=2048, metavar='IB', help='Imitation learning minibatch size')
parser.add_argument('--imitation-replay-size', type=int, default=4, metavar='IRS', help='Imitation learning trajectory replay size')
parser.add_argument('--r1-reg-coeff', type=float, default=1, metavar='γ', help='R1 gradient regularisation coefficient')



args = parser.parse_args()




def eachFile(filepath):
    dir_list=[]
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        dir_list.append(child)
    return dir_list




def main():
    '''
    exp_logdir=eachFile('D:/swarm_ppo2/swarm_ppo/PPO_DATA/')




    
 
    data_num_list=[50]

    agent_num_list=[10]
    para_list=[]
    for a,b in itertools.product(agent_num_list,data_num_list):
        para_list.append([a,b])
    for i in range(len(exp_logdir)):
        para_list[i].append(exp_logdir[i])
    

    for item in para_list:
        exp_logdir=item[2]
        n_agents=item[0]
        exp_len=item[1]
        learn.airl_learn(exp_logdir,n_agents,exp_len,args.seed,args.steps,args.hidden_size,args.discount,args.trace_decay,args.ppo_clip,args.ppo_epochs,args.value_loss_coeff,args.entropy_loss_coeff,args.learning_rate,args.batch_size,args.imitation,args.state_only,\
          args.imitation_epochs,args.imitation_batch_size,args.imitation_replay_size,args.r1_reg_coeff)
    '''
    data_num_list=[50]

    agent_num_list=[10]
    exp_logdir='D:/swarm_ppo2/swarm_ppo/PPO_DATA/'
  
    for a,b in itertools.product(agent_num_list,data_num_list):
        exp_logdir=os.path.join(exp_logdir, 'na_{}-epi_{}-obsm_{}'.format(a, b, 'fix_acc')+'vicsek_data.pth')
        learn.airl_learn(exp_logdir,a,b,args.seed,args.steps,args.hidden_size,args.discount,args.trace_decay,args.ppo_clip,args.ppo_epochs,args.value_loss_coeff,args.entropy_loss_coeff,args.learning_rate,args.batch_size,args.imitation,args.state_only,\
          args.imitation_epochs,args.imitation_batch_size,args.imitation_replay_size,args.r1_reg_coeff)
    
    
    
    
if __name__ == "__main__":
    main()