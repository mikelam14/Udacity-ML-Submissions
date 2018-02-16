# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:27:45 2017

@author: pc
"""
import matplotlib.pyplot as plt
import os, ast, pandas as pd, numpy as np, time
os.chdir('C:\\Users\\pc\\Google Drive (lam.yanpui14@gmail.com)\\Mikee\\Self Learn\\Udacity\\ML nanodegree\\ML-Submissions\\smartcab\\smartcab')
import agent as a

os.chdir('C:\\Users\\pc\\Google Drive (lam.yanpui14@gmail.com)\\Mikee\\Self Learn\\Udacity\\ML nanodegree\\ML-Submissions\\smartcab')
import visuals as vs

f = 'sim_improved-learning.csv'

def getTestRatings(csv):
    data = pd.read_csv(os.path.join("logs", csv))
    testing_data = data[data['testing'] == True].copy()
    
    # safety (rate)
    testing_data['good_actions'] = testing_data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    good_ratio = \
    testing_data['good_actions'].sum() * 1.0 / (testing_data['initial_deadline'] - testing_data['final_deadline']).sum()
    
    # safety (grade)
    if good_ratio == 1: # Perfect driving
        safeRate = "A+"
    else: # Imperfect driving
        if testing_data['actions'].apply(lambda x: ast.literal_eval(x)[4]).sum() > 0: # Major accident
            safeRate = "F"
        elif testing_data['actions'].apply(lambda x: ast.literal_eval(x)[3]).sum() > 0: # Minor accident
            safeRate = "D"
        elif testing_data['actions'].apply(lambda x: ast.literal_eval(x)[2]).sum() > 0: # Major violation
            safeRate = "C"
        else: # Minor violation
            minor = testing_data['actions'].apply(lambda x: ast.literal_eval(x)[1]).sum()
            if minor >= len(testing_data)/2: # Minor violation in at least half of the trials
                safeRate = "B"
            else:
                safeRate = "A"
    
    # reliability (ratings)
    success_ratio = testing_data['success'].sum() * 1.0 / len(testing_data)
    
    # relibaility (grades)
    if success_ratio == 1: # Always meets deadline
        reliRate = "A+"
    else:
        if success_ratio >= 0.90:
            reliRate = "A"
        elif success_ratio >= 0.80:
            reliRate = "B"
        elif success_ratio >= 0.70:
            reliRate = "C"
        elif success_ratio >= 0.60:
            reliRate = "D"
        else:
            reliRate = "F"
    
    # avg reward of testing
    avg_reward = np.mean(testing_data['net_reward'])
    
    # ending deadline
    avg_deadline = np.mean(testing_data['final_deadline'] * 1.0 / testing_data['initial_deadline'])
    
    # avg states visited
    avg_states = np.mean(testing_data['states_visited'])
    
    # avg zero Q-value counts (zero Q_value means we are in that state but we never try that action)
    avg_zCount = np.mean(testing_data['zCount']*1.0 / testing_data['states_visited'])
    
    return(good_ratio, success_ratio, safeRate, reliRate, avg_reward, avg_deadline, avg_states, avg_zCount)

# run many simulations
safe, reliability, safeRate, reliRate, reward, deadline, states, zCount = [],[],[],[],[],[],[],[]
n = 1
start = time.time()
for i in range (n):
    a.run()
    # 1. read file and get safety and reliability ratings
    s, r, sr, rr, rew, d, st, z = getTestRatings(f)
    safe.append(s)
    reliability.append(r)
    safeRate.append(sr)
    reliRate.append(rr)
    reward.append(rew)
    deadline.append(d)
    states.append(st)
    zCount.append(z)

print('Finished {} sim in {} minutes. Safety: {}, reliability: {}, reward: {}, deadline: {}, states: {}, zCount: {}'\
      .format(n,round((time.time()-start)*1.0/60,3),np.mean(safe),np.mean(reliability),
              np.mean(reward),np.mean(deadline),np.mean(states),np.mean(zCount)))

s_values, s_counts = np.unique(safeRate, return_counts=True)
print('Safety: ',s_values,s_counts)

r_values, r_counts = np.unique(reliRate, return_counts=True)
print('Reliability: ',r_values,r_counts)

vs.plot_trials(f)
    
    