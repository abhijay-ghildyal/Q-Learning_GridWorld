#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:28:36 2018

@author: abhijay
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
sys.setrecursionlimit(20000)

import pandas as pd

import matplotlib.pyplot as plt

class Env:
    def __init__(self):
        ##### Environment intializations #####
        self.gridWorld = np.zeros((10,10))
        self.wall = [ [3,2], [3,3], [3,4], [3,5], [3,7], [3,8], [3,9], [4,5], [5,5], [6,5], [7,5], [8,5]]
        self.reward = {(4,4):-1, (5,6):-1, (5,7):-1, (6,6): 1, (6,7): -1, (6,9): -1, (7,9): -1, (8,4): -1, (8,6):-1, (8,7):-1}
        self.goal = [6,6]   
        
class Q_learning_Agent:
    def __init__(self):
        ##### Agent intializations #####
        self.beta = 0.9
        self.actions = {'L':1,'R':2,'U':3,'D':4}
        self.actions_ = ['L','R','U','D']
        self.env = Env()
        self.epsilon = 0.1 # Change
        self.alpha = 0.01
        self.q = np.zeros((10,10,4))
        self.nVisits = np.zeros((10,10))
        # self.nReplays = 5000
        self.startTemp = 5
    
    def init_agent(self):
        # Assuming the top left corner to be [1,1]
        self.currentLocation = [1,1]
        self.totalReward = 0
        
    def get_action(self, exploration='e-greedy'):
        
        if exploration == 'e-greedy':
            ##### Epsilon greedy exploration #####
            if self.epsilon > np.random.random():
                # Explore
                action = self.get_move_eGreedy(greedy=False)
            else:
                # Exploit
                action = self.get_move_eGreedy()
        else:
            ##### Boltzmann exploration #####
            action = self.get_move_boltzmann()
        
        return action
    
    ##### This function returns location of the new state based on the action taken #####
    def get_new_state(self, action):
        
        new_state = self.currentLocation[:]
        
        if action == self.actions['L']:
            new_state[1] -= 1
        elif action == self.actions['R']:
            new_state[1] += 1
        elif action == self.actions['U']:
            new_state[0] -= 1
        elif action == self.actions['D']:
            new_state[0] += 1
            
        return new_state
    
    ##### This function makes the agent play in the environment #####    
    def act(self, exploration='e-greedy'):
        
        ##### nVisits is used in deciding temp in Boltzman exploration #####
        self.nVisits[self.currentLocation[0]-1,self.currentLocation[1]-1]+=1
        
        ##### Get the action taken by agent #####
        action = self.get_action(exploration)
        
        state = [self.currentLocation[0],self.currentLocation[1]]
        state_ = self.get_new_state(action+1) ##### Next state #####
        
        if state_ != self.env.goal:
            self.currentLocation = state_
            self.act()
        
        if (state_[0], state_[1]) in self.env.reward.keys():
            reward = self.env.reward[(state_[0], state_[1])]
        else:
            reward = 0
            
        self.totalReward+=reward
            
        maxQnext = np.max(self.q[ state_[0]-1, state_[1]-1])
        
        # Q learning
        self.q[ state[0]-1, state[1]-1, action] += \
                self.alpha * ( reward + self.beta * maxQnext - \
                                  self.q[ state[0]-1, state[1]-1, action])
        
    
    ##### Assuming that the agent is always facing upwards #####
    def get_move_eGreedy(self, greedy=True):
        
        moves = self.valid_moves()
        
        if greedy:            
            maxValue = max(self.q[ self.currentLocation[0]-1, self.currentLocation[1]-1, moves])
            
            ##### In the beginning it's actions tend to oscillate. This breaks the tie when argmax(action) are all equal and 0 #####
            if maxValue == 0:
            	moves = [moves[i] for i in np.where(self.q[ self.currentLocation[0]-1, self.currentLocation[1]-1, moves]==0)[0]]
            return np.random.choice(moves)
        else:
            return moves[np.random.randint(0, len(moves))]
    
    def get_move_boltzmann(self):
        
        ##### Decide temp based on visits in the state #####
        T = self.startTemp / self.nVisits[self.currentLocation[0]-1,self.currentLocation[1]-1]
        
        moves = self.valid_moves()
        
        Qvalues = self.q[ self.currentLocation[0]-1, self.currentLocation[1]-1, moves]
        
        prob = np.exp(Qvalues/T)/sum(np.exp(Qvalues/T))
        
        ##### Select best move based on probability #####
        return moves[np.argmax(prob)]
        
    def valid_moves(self):
        ##### Returns valid moves that can be taken by the agent #####

        moves = []
        
        if self.currentLocation[1]>1 and [self.currentLocation[0],self.currentLocation[1]-1] not in self.env.wall:
            moves.append(self.actions['L']-1)
        
        if self.currentLocation[1]<10 and [self.currentLocation[0],self.currentLocation[1]+1] not in self.env.wall:
            moves.append(self.actions['R']-1)
        
        if self.currentLocation[0]>1 and [self.currentLocation[0]-1,self.currentLocation[1]] not in self.env.wall:
            moves.append(self.actions['U']-1)
        
        if self.currentLocation[0]<10 and [self.currentLocation[0]+1,self.currentLocation[1]] not in self.env.wall:
            moves.append(self.actions['D']-1)
        
        return moves
    
    def get_current_policy(self):
        ##### Generates the policy #####

        policy = np.zeros((10,10))
        for i in range(self.q.shape[0]):
            for j in range(self.q.shape[1]):
                if [i+1,j+1] in self.env.wall:
                    policy[i,j] = 0
                else:                    
                    policy[i,j] = np.argmax(self.q[i,j,:])
        return policy
    
    def display_grid(self):
        for i in range(self.q.shape[0]):
            for j in range(self.q.shape[1]):
                if [i+1,j+1] in self.env.wall:
                    print ('  w   |',end='')
                elif [i+1,j+1] == self.env.goal:
                    print (' g=1  |', end='')
                elif (i+1,j+1) in self.env.reward.keys():
                    print (str('%.3f' % self.env.reward[(i+1,j+1)])+"|", end='')
                else:                         
                    maxVal = max(self.q[i,j,:])
                    if maxVal>=0:
                        print (str(' %.3f' % round(maxVal,3))+"|", end='')
                    else:
                        print (str('%.3f' % round(maxVal,3))+"|", end='')
            print ()
    
    def action_symbol(self, action):
        if action == self.actions['L']:
            return '<'
        elif action == self.actions['R']:
            return '>'
        elif action == self.actions['U']:
            return '^'
        elif action == self.actions['D']:
            return 'v'
    
    def check_convergence(self, prev_policy):
        policy = self.get_current_policy()
        return np.sum((policy-prev_policy)**2)
            
    def display_policy(self):
        for i in range(self.q.shape[0]):
            for j in range(self.q.shape[1]):
                if [i+1,j+1] in self.env.wall:
                    print ('  w  |',end='')
                elif [i+1,j+1] == self.env.goal:
                    print (' g=1 |', end='')
                elif (i+1,j+1) in self.env.reward.keys():
                    maxAction = np.argmax(self.q[i,j,:])
                    print (str(' %.0f' % self.env.reward[(i+1,j+1)])+','+self.action_symbol(maxAction+1)+"|", end='')
                else:                         
                    maxAction = np.argmax(self.q[i,j,:])
                    print ('   '+self.action_symbol(maxAction+1)+' |',end='')
            print ()
    
    def plot_figure( self, Values, y_limit, title, y_label, x_label, saveAs):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot( Values)
        ax.set_ylim(y_limit[0], y_limit[1])
        ax.set_title( title, fontsize=18)
        ax.set_ylabel( y_label, fontsize=15)
        ax.set_xlabel( x_label, fontsize=15)
        fig.savefig('plots/'+saveAs+'.png')
        
if __name__ == "__main__":
    
    print ("===== Q9 Qlearning for Gridworld =====\n")
    print ("For each experiment it runs 20,000 iterations")
    print ("Printing results at every 5000 iterations\n")
    experiments = [['e-greedy',0.1],['e-greedy',0.2],['e-greedy',0.3],['boltzmann',1],['boltzmann',5],['boltzmann',10],['boltzmann',20]]
    # experiments = [['e-greedy',0.1],['boltzmann',5]]
    for experiment in experiments:
        print ("\n===== Experiment: "+str(experiment)+" =====\n")
        agent = Q_learning_Agent()
        if experiment[0]=='e-greedy':
            agent.epsilon = experiment[1]
        else:
            agent.startTemp = experiment[1]
            
        rewardsPerEpisode = []
        policyDiff = []
        itr=0
        while True:
            agent.init_agent()
            itr+=1
            
            agent.act(exploration=experiment[0])
            
            if itr > 1:
                policyDiff.append(agent.check_convergence(prev_policy))
            
            prev_policy = agent.get_current_policy()
            
            if itr%5000==0:
                print("\n Iteration:"+str(itr))
                agent.display_grid()
                agent.display_policy()
            if itr == 20000:
                agent.plot_figure(policyDiff, [min(policyDiff)-2,max(policyDiff)+2], "Change in Policy per Episode (squared change), Experiment:"+str(experiment), "Change (squared)", "Episode", "Convergence, Experiment:("+str(experiment[0])+"_"+str(experiment[1])+")")
                # agent.plot_figure(rewardsPerEpisode, [-250,1.5], "Rewards per Episode, Experiment:"+str(experiment), "Rewards", "Episode", "Rewards per Episode, Experiment:("+str(experiment[0])+"_"+str(experiment[1])+")")
                
                # rMean=pd.Series.rolling( pd.Series(rewardsPerEpisode), window=10).mean()
                # rMean=rMean.fillna(0)
                # agent.plot_figure(rMean, [min(rMean)-5,max(rMean)+5], "MA of Rewards per Episode, Experiment:"+str(experiment), "Avg. of previous 10 Total episode rewards", "Episode", "10 MA Rewards per Episode, Experiment:("+str(experiment[0])+"_"+str(experiment[1])+")")
                
                # rMean=pd.Series.rolling( pd.Series(rewardsPerEpisode), window=50).mean()
                # rMean=rMean.fillna(0)
                # agent.plot_figure(rMean, [min(rMean)-5,max(rMean)+5], "MA of Rewards per Episode, Experiment:"+str(experiment), "Avg. of previous 50 Total episode rewards", "Episode", "50 MA Rewards per Episode, Experiment:("+str(experiment[0])+"_"+str(experiment[1])+")")
                
                # rMean=pd.Series.rolling( pd.Series(rewardsPerEpisode), window=100).mean()
                # rMean=rMean.fillna(0)
                # agent.plot_figure(rMean, [min(rMean)-5,max(rMean)+5], "MA of Rewards per Episode, Experiment:"+str(experiment), "Avg. of previous 100 Total episode rewards", "Episode", "100 MA Rewards per Episode, Experiment:("+str(experiment[0])+"_"+str(experiment[1])+")")
                
                # rMean=pd.Series.rolling( pd.Series(rewardsPerEpisode), window=1000).mean()
                # rMean=rMean.fillna(0)
                # agent.plot_figure(rMean, [min(rMean)-5,max(rMean)+5], "MA of Rewards per Episode, Experiment:"+str(experiment), "Avg. of previous 1000 Total episode rewards", "Episode", "1000 MA Rewards per Episode, Experiment:("+str(experiment[0])+"_"+str(experiment[1])+")")
                
                break
            rewardsPerEpisode.append(agent.totalReward)
        
    