#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        # TO DO: Initialize relevant elements
        self.Q_sa = np.zeros((n_states,n_actions)) # Q-value table
        self.n_sa_s = np.zeros((n_states,n_actions,n_states)) # Transition count table
        self.Rsum_sa_s = np.zeros((n_states,n_actions,n_states)) # Reward sum table

    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        if np.random.uniform() < epsilon:
            a = np.random.randint(0,self.n_actions) # random action
        else:
            a = np.argmax(self.Q_sa[s]) # greedy action selection
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # TO DO: Add Dyna update
        self.n_sa_s[s,a,s_next] += 1
        self.Rsum_sa_s[s,a,s_next] += r

        if done:
            td_target = r
        else:
            td_target = r + self.gamma * np.max(self.Q_sa[s_next])

        self.Q_sa[s,a] += self.learning_rate * (td_target - self.Q_sa[s,a])

        visited = np.argwhere(self.n_sa_s.sum(axis=2) > 0)

        if len(visited) == 0:
            return
        
        for _ in range(n_planning_updates):
            idx = np.random.randint(len(visited))
            sim_s, sim_a = visited[idx]

            counts = self.n_sa_s[sim_s, sim_a]
            prob = counts / counts.sum()
            sim_s_next = np.random.choice(self.n_states, p=prob)

            sim_r = self.Rsum_sa_s[sim_s, sim_a, sim_s_next] / self.n_sa_s[sim_s, sim_a, sim_s_next]

            sim_target = sim_r + self.gamma * np.max(self.Q_sa[sim_s_next])
            self.Q_sa[sim_s, sim_a] += self.learning_rate * (sim_target - self.Q_sa[sim_s, sim_a])




        pass

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # TO DO: Initialize relevant elements
        self.Q_sa = np.zeros((n_states,n_actions))
        self.n_sa_s = np.zeros((n_states,n_actions,n_states)) 
        self.Rsum_sa_s = np.zeros((n_states,n_actions,n_states))
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        if np.random.uniform() < epsilon:
            a = np.random.randint(0,self.n_actions) 
        else:
            a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        
        # TO DO: Add Prioritized Sweeping code
        self.n_sa_s[s,a,s_next] += 1
        self.Rsum_sa_s[s,a,s_next] += r

        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a))) 
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue

        if done:
            p = abs(r - self.Q_sa[s,a])
        else:
            p = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s,a])

        if p > self.priority_cutoff:
            self.queue.put((-p,(s,a)))
            
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break

            _, (sim_s, sim_a) = self.queue.get()

            counts = self.n_sa_s[sim_s, sim_a]
            if counts.sum() == 0:
                continue
            prob = counts / counts.sum()
            sim_s_next = np.random.choice(self.n_states, p=prob)

            sim_r = self.Rsum_sa_s[sim_s, sim_a, sim_s_next] / self.n_sa_s[sim_s, sim_a, sim_s_next]

            sim_target = sim_r + self.gamma * np.max(self.Q_sa[sim_s_next])
            self.Q_sa[sim_s, sim_a] += self.learning_rate * (sim_target - self.Q_sa[sim_s, sim_a])

            for s_bar in range(self.n_states):
                for a_bar in range(self.n_actions):
                    if self.n_sa_s[s_bar, a_bar, sim_s] > 0:
                        r_bar = self.Rsum_sa_s[s_bar, a_bar, sim_s] / self.n_sa_s[s_bar, a_bar, sim_s]
                        p_bar = abs(r_bar + self.gamma * np.max(self.Q_sa[sim_s]) - self.Q_sa[s_bar, a_bar])
                        if p_bar > self.priority_cutoff:
                            self.queue.put((-p_bar, (s_bar, a_bar)))
        pass

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'ps' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = False
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = True
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    print("Done!")
    print(f"Max Q-value: {pi.Q_sa.max():.2f}")
    print(f"Mean Q-value: {pi.Q_sa.mean():.2f}")

if __name__ == '__main__':
    test()
