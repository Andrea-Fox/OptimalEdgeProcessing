import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import math
import random

class BatteryBuffer():

    def __init__(self, M, B, h, p_z, cost_probability, reward_function, disadvantage_function = None, disadvantage_function_exponent = None, gamma = 0.9, eps_0 = 0, exploration_rate = 0, mean_value_normal_distr = 3, sigma_normal_distr = 3, lambda_harvesting_distribution = 1, delta = 1) :
        self.num_states = int(M*(B+1)*2)
        self.M = int(M)
        self.B = int(B)
        self.h = h
        self.p_z = p_z              # probability that the server is available
        self.eps0 = eps_0
        self.reward_function = reward_function
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.delta = delta

        # disadvantage_function = (lambda e: (B-e)**1.1)
        self.disadvantage_function_exponent = disadvantage_function_exponent
        self.disadvantage_function = disadvantage_function
        
        # definition of the cost distribution
        energy_span=int(B+1);                 # range of energy consumption

        p_C=np.zeros((energy_span+2, ))              # vector of energy cost probabilities (0,1,...,B+h)
        
        if cost_probability == 1:
            # uniform distribution of cost 
            p_C[1:(B+2)]=1/(energy_span)       # uniform probability on cost (only ocnsidering costs between 1 and B+h)   
        elif cost_probability == 2:
            p_C[1:] = self.eps0 * np.ones(energy_span+1)
            p_C[1] = 0.5
            p_C[B+1] = 0.5
            p_C = p_C/sum(p_C)           
        elif cost_probability == 3:
            # normal distribution
            # mean_value = 3 # (B+1+h)/2
            for c in range(1, energy_span +2):
                p_C[c] = 1/(sigma_normal_distr * math.sqrt(2*math.pi)) *  math.exp(-0.5*(((c) - mean_value_normal_distr)/sigma_normal_distr)**2)
            # p_C[5] = 1
            p_C = p_C/sum(p_C)
        elif cost_probability == 4:
            # deterministic distribution of cost: works
            p_C[1:] = self.eps0*np.ones(energy_span+1)
            constant_cost_element = 5                  # value of cost that will have probability equal to 1 
            p_C[constant_cost_element]= 1;              # constant cost for a certain value of energy 
            p_C = p_C/sum(p_C)
        elif cost_probability == 5:
            # poisson distribution
            lamb = 2
            for c in range(1, energy_span+1):
                p_C[c] = lamb**(c) * math.exp(-lamb) / math.factorial(c)
            p_C = p_C/sum(p_C)
        
        # print("Cost distribution = " + str(p_C))
        self.p_C = p_C
        
        self.cumulative_distribution = np.zeros(energy_span+1)
        for i in range(1, energy_span+1):
            self.cumulative_distribution[i] = self.cumulative_distribution[i-1] + self.p_C[i]

        self.p_H = np.zeros(energy_span+2)
        # 
        if lambda_harvesting_distribution == 0:
            # harvesting rate is always one (we chose that) 
            self.p_H[1] = 1
        else:
            lamb = lambda_harvesting_distribution
            for h in range(1, energy_span+1):
                self.p_H[h] = lamb**(h) * math.exp(-lamb) / math.factorial(h)
            self.p_H = self.p_H/sum(self.p_H)
        
        self.max_episodes_steps = 100
        return

    # step function when h is stochastic
    def step_stochastic_h(self, state, action):
        # obtain energy, aoi and server availability from state index
        aoi, energy, server_available = self.compute_state_values(state)

        success = False

        if server_available:
            # action is always 1
            new_aoi = 1
            harvesting_rate = random.choices(np.arange(len(self.p_H)), weights = self.p_H)[0]
            new_energy = min(energy + harvesting_rate, self.B)
            reward = self.reward_function[min(aoi-1+self.delta, self.M-1)]
            success = True
        else:
            if action == 1:
                # we do the action: sample a cost, if it is lower than the energy available, then we can conclude the action, otherwise it fails
                cost_of_action = random.choices(np.arange(len(self.p_C)), weights = self.p_C)[0]
                harvesting_rate = random.choices(np.arange(len(self.p_H)), weights = self.p_H)[0]
                if cost_of_action <= energy + harvesting_rate:
                    # action completed
                    reward = self.reward_function[aoi-1] # - self.disadvantage_function(energy + self.h - cost_of_action)
                    new_aoi = 1
                    new_energy = min(self.B, energy + harvesting_rate - cost_of_action)
                    success = True
                else:
                    new_aoi = 1
                    reward = self.reward_function[aoi - 1] - self.disadvantage_function(energy + harvesting_rate - cost_of_action)
                    # we need to do a number of harvestings sufficient to go back to a positive energy
                    new_energy = energy + harvesting_rate - cost_of_action
                    # new_energy is negative
                    while new_energy < 0:
                        new_harvesting = random.choices(np.arange(len(self.p_H)), weights = self.p_H)[0]
                        new_energy += new_harvesting
                    
            elif action == 0:
                new_aoi = min(aoi + 1, self.M)
                harvesting_rate = random.choices(np.arange(self.B + 3), weights = self.p_H)[0]
                new_energy = min(energy + harvesting_rate, self.B)
                reward = self.reward_function[aoi - 1]
        if self.p_z >=1:
            new_server_available = 1
        elif self.p_z == 0 : 
            new_server_available = 0
        else:
            new_server_available = random.choices([0, 1], weights=[1-self.p_z, self.p_z], k = 1 )[0]

        # computation of next_state index
        next_state = new_server_available*(self.M*(self.B+1)) + (new_aoi-1)*(self.B+1) + new_energy
        return next_state, reward, success


    # step function when h is constant 
    def step(self, state, action):

        # obtain energy, aoi and server availability from state index
        server_available = state // (self.M*(self.B+1))
        state = state - server_available*self.M*(self.B+1)

        aoi = (state // (self.B+1)) +1
        energy = state % (self.B +1)

        done = False
        success = False

        if server_available:
            # action is always 1
            new_aoi = 1
            new_energy = min(energy + self.h, self.B)
            reward = self.reward_function[min(aoi-1+self.delta, self.M-1)]
            success = True
        else:
            if action == 1:
                # we do the action: sample a cost, if it is lower than the energy available, then we can conclude the action, otherwise it fails
                cost_of_action = np.random.choice(np.arange(self.B + self.h +2), p = self.p_C)
                if cost_of_action <= energy + self.h:
                    # action completed
                    reward = self.reward_function[aoi-1] # - self.disadvantage_function(energy + self.h - cost_of_action)
                    new_aoi = 1
                    new_energy = min(self.B, energy + self.h - cost_of_action)
                    done = True
                    success = True
                else:
                    new_aoi = 1
                    new_energy = 0
                    reward = self.reward_function[aoi - 1] - self.disadvantage_function(energy + self.h - cost_of_action) 
                    done = True
            elif action == 0:
                new_aoi = min(aoi + 1, self.M)
                new_energy = min(energy + self.h, self.B)
                reward = self.reward_function[aoi - 1]

        new_server_available = np.random.binomial(1, self.p_z)

        # computation of next_state index
        # next_state = new_server_available*(self.M*(self.B+1)) + (new_aoi-1)*(self.B+1) + new_energy
        next_state = self.compute_index(new_aoi, new_energy, new_server_available)
        return next_state, reward, done

    def reset(self):
        energy = np.random.choice(self.B+1)
        server_available = np.random.binomial(1, self.p_z)
        # aoi = 1
        initial_state = server_available*(self.M*(self.B+1)) + energy

        return initial_state

    def reset_all_states(self):
        energy = np.random.choice(self.B+1)
        server_available = np.random.binomial(1, self.p_z)
        aoi = np.random.choice(self.M) +1

        return server_available*(self.M*(self.B+1)) + (aoi-1)*(self.B+1) + energy
    
    def reset_edge_states(self, theta):
        energy = np.random.choice(self.B + 1)
        server_available = np.random.binomial(1, self.p_z)
        aoi = math.floor(theta[energy]) + 1
        return server_available*(self.M*(self.B+1)) + (aoi-1)*(self.B+1) + energy  

    def reset_all_recurrent_states(self):
        energy = np.random.choice(self.B + 1)
        server_available = np.random.binomial(1, self.p_z)
        if energy > 0:
            aoi = np.random.choice(energy) + 1
        else:
            aoi = 1
        return server_available*(self.M*(self.B+1)) + (aoi-1)*(self.B+1) + energy  


    def compute_state_values(self, state):
        
        server_available = state //  (self.M*(self.B+1))
        
        state = state - server_available*self.M*(self.B+1)

        aoi = (state // (self.B+1)) +1
        energy = state % (self.B +1)

        return aoi, energy, server_available


    def compute_index(self, aoi, energy, server_available):
        return server_available * self.M *(self.B+1) + (aoi-1)*(self.B+1) + energy


    def policy(self, state, theta):
        aoi, energy, server_available = self.compute_state_values(state)
        
        if server_available == 1 or energy == self.B:
            return 1
        else:
            if aoi <= math.floor(theta[energy]):
                return 1 - self.exploration_rate
            elif aoi == math.ceil(theta[energy]):
                return theta[energy] - math.floor(theta[energy])
            else:
                return self.exploration_rate


    def policy_der(self, state, theta):
        
        aoi, energy, server_available = self.compute_state_values(state)

        if server_available == 1 or energy == self.B:
            return 0
        else:
            if aoi <= math.floor(theta[energy]):
                return 0
            elif aoi == math.ceil(theta[energy]):
                return - 1
            else:
                return 0

