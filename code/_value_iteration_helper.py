import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np

import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def compute_index(x, e, server_availability, M, B):
    return server_availability*(M*(B+1)) + (x-1)*(B+1) + e

def server_probability(server_availability, p_z):
    return p_z*server_availability+(1-p_z)*(1-server_availability)

def R(x, v, reward_function, M, delta = 1):
    x_eff=min([x+v*delta, M])# *np.ones(1,len(x))])
    return reward_function[x_eff-1]

def plot_solution(results, M, B, algorithm = '', steps_number = -1,cost_probability = -1, p_z = -1):
    tile_color = dict(  action_wait=[222/256, 41/256, 41/256],
                    action_read = [41/256, 222/256, 71/256], 
                    action_wait_transient = [239/256, 148/256, 148/256],
                    action_read_transient = [148/256, 239/256, 163/256])

    grid = tile_color['action_read'] * np.ones((B+1,M, 3))
    # grid[:, 0] = tile_color['action_wait']


    for x in range(1, M+1):
        for e in range(B+1):
            if x <= e+1:
                
                if results[x-1,e]== 0:
                    # optimal action is 0 (wait)
                    grid[B-e, x-1, :] = tile_color['action_wait']
                else:
                    grid[B-e, x-1, :] = tile_color['action_read']
            else:
                if results[x-1,e]== 0:
                    # optimal action is 0 (wait)
                    grid[B-e, x-1, :] = tile_color['action_wait_transient']
                else:
                    grid[B-e, x-1, :] = tile_color['action_read_transient']

    plt.figure()
    im = plt.imshow(grid, interpolation='none', vmin=0, vmax=1, aspect='equal')

    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, M, 1))
    ax.set_yticks(np.arange(0, B+1, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, M+1, 1))
    ax.set_yticklabels(np.arange(B, -1, -1))

    ax.set_xticks(np.arange(-0.5, M, 1), minor = True)
    ax.set_yticks(np.arange(-0.5, B, 1), minor = True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.set_title("Optimal policy", weight='bold', fontsize=18)

    plt.xlabel('Age of Information', fontsize = 14)
    plt.ylabel('Battery level', fontsize = 14)

    t = 0.75
    cmap = {1:[222/256, 41/256, 41/256,t],2:[41/256, 222/256, 71/256,t]}
    labels = {1:'wait',2:'read'}    
    ## create patches as legend
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]

    plt.rcParams.update({'font.size': 13})    
    plt.legend(handles=patches, loc=1, borderaxespad=0.)

    
def compute_P0(env, harvesting_distribution):
    num_states = (env.B+1) * env.M * 2
    P0 = np.zeros((env.M*(env.B+1), num_states))
    for x1 in range(1, env.M+1):
        for e1 in range(env.B+1):
            for x2 in range(1, env.M+1):
                for e2 in range(env.B+1):                               
                    for v2 in range(2):      
                        if((x2==min(x1+1,env.M)) and (e2 > e1) and e2 < env.B):     
                            index_s1 = compute_index(x1,e1,0, env.M, env.B)               
                            index_s2 = compute_index(x2,e2,v2, env.M, env.B)               
                            P0[index_s1,index_s2] = server_probability(v2, env.p_z)*harvesting_distribution[e2 -  e1]
                        elif ((x2==min(x1+1,env.M)) and (e2 >= e1) and e2 == env.B):
                            index_s1 = compute_index(x1,e1,0, env.M, env.B)               
                            index_s2 = compute_index(x2,e2,v2, env.M, env.B)               
                            P0[index_s1,index_s2] = server_probability(v2, env.p_z)*np.sum(harvesting_distribution[(env.B - e1):])
    return P0

def compute_matrix_gamma(env):
    max_support_C = len(env.p_C)
    gamma_matrix = np.zeros((max_support_C, max_support_C * env.B))
    # case k = 1
    for r in range(1, env.B+1):
        gamma_matrix[1, r] = env.p_H[r]
    for k in range(2, max_support_C):
        for r in range(k, k*env.B +1):
            for j in range(1, env.B):
                gamma_matrix[k, r] += gamma_matrix[k-1, r-j] * env.p_H[j]
    return gamma_matrix



def compute_P1(env, cost_distribution, harvesting_distribution):
    num_states = (env.B+1) * env.M * 2
    gamma_matrix = compute_matrix_gamma(env)
    P1=np.zeros((num_states, num_states))
    for x1 in range(1, env.M+1):
        for e1 in range(env.B+1):
            for v1 in range(2):
                for x2 in range(1, env.M+1):
                    for e2 in range(env.B+1):                               
                        for v2 in range(2):                                   
                            index_s1=compute_index(x1,e1,v1, env.M, env.B)                  
                            index_s2=compute_index(x2,e2,v2, env.M, env.B)                  
                            if(x2==1):
                                if(v1==0):  
                                    if (e2 < env.B):             # action succeeds and final energy between 0 and B
                                        # we suceed at the first attempt
                                        convolution = 0
                                        for t in range(max(1, 1 + e2 - e1), min(env.B+2, env.B+2+e2-e1)+1):
                                            convolution += harvesting_distribution[t] * cost_distribution[e1 + t - e2]
                                        P1[index_s1, index_s2] = convolution
                                        # we now need to add the term indicating the probability of getting to this energy level after k harvesting
                                        for k in range(2, len(env.p_C)):
                                            for r in range(k-1, env.B * (k-1)):
                                                for c in range(e1+r+1, len(env.p_C)):
                                                    try:
                                                        P1[index_s1, index_s2] += gamma_matrix[k-1, r] * env.p_C[c] * env.p_H[e2 + c - r - e1]
                                                    except:
                                                        a = 0
                                        P1[index_s1, index_s2] = server_probability(v2, env.p_z) * P1[index_s1, index_s2]
                                    elif e2 == env.B:           # the action has succeeded and the final energy level is B
                                        exceeding_energy_probability = 0
                                        for t in range(env.B - e1 + 1, len(harvesting_distribution)):
                                            for k in range(1, e1 + t - env.B +1):
                                                exceeding_energy_probability += harvesting_distribution[t] * cost_distribution[k]
                                        P1[index_s1, index_s2] = server_probability(v2, env.p_z) * exceeding_energy_probability
                                else: 
                                    if((e2 > e1) and e2 < env.B):     
                                        index_s1 = compute_index(x1,e1,1, env.M, env.B)               
                                        index_s2 = compute_index(x2,e2,v2, env.M, env.B)               
                                        P1[index_s1,index_s2] = server_probability(v2, env.p_z)*harvesting_distribution[e2 -  e1]
                                    elif ((e2 >= e1) and e2 == env.B):
                                        index_s1 = compute_index(x1,e1,1, env.M, env.B)               
                                        index_s2 = compute_index(x2,e2,v2, env.M, env.B)               
                                        P1[index_s1,index_s2] = server_probability(v2, env.p_z)*np.sum(harvesting_distribution[(env.B - e1):])

    return P1


def compute_optimal_solution(env, disadvantage_function, reward_function, cost_distribution, harvesting_distribution, random_seed = -1, return_v_estimate = False):
    
    if random_seed>0:
        random.seed(random_seed)
    
    num_states = env.M*(env.B+1)*2
    cumulative_distribution = np.zeros(env.B+1+1, )
    for i in range(1, env.B+1+1):
        cumulative_distribution[i] = cumulative_distribution[i-1] + cost_distribution[i]

    # definition of the transition matrixes
    # transition probabilities when the action selected is a=0
    P0 = compute_P0(env, harvesting_distribution=harvesting_distribution)


    # Transition probabilities when a=1 "process"
    P1 = compute_P1(env, cost_distribution = cost_distribution, harvesting_distribution=harvesting_distribution)



    # we check if the the transition probabilities have sum 1 on the rows
    # for row_index in range(P0.shape[0]):
    #     print(sum(P1[row_index, :]))


    # definition of the reward
    Rew1 = np.zeros((num_states, ))
    Rew0 = np.zeros((env.M*(env.B+1), ))

    beta = 1
    for x in range(1, env.M+1):
        for e in range(env.B+1):
            index=compute_index(x,e,0, env.M, env.B)
            # print(disadvantage_function)
            try: 
                if isinstance(disadvantage_function.dtype, 'float64'):

                    disadvantage = 0
                    fix_indexes = lambda x: -x*(x<0)
                    for h in range(1, len(harvesting_distribution)):
                        for c in range(1, len(cost_distribution)):
                            disadvantage += harvesting_distribution[h]*cost_distribution[c]*disadvantage_function[fix_indexes(e + h - c)]

                    Rew1[index] = R(x, 0, reward_function, env.M, env.delta) - beta * disadvantage
            except:
                disadvantage = 0
                fix_indexes = lambda x: x*(x<0)
                for h in range(1, len(harvesting_distribution)):
                    for c in range(1, len(cost_distribution)):
                        disadvantage += harvesting_distribution[h]*cost_distribution[c]*disadvantage_function(fix_indexes(e + h - c))
                Rew1[index] = R(x, 0, reward_function, env.M, env.delta) - beta * disadvantage
            
            Rew0[index] = R(x, 0, reward_function, env.M, env.delta)
            index=compute_index(x,e,1, env.M, env.B)
            Rew1[index] = R(x,1, reward_function, env.M, env.delta)


    # value iteration algorithm
    mu_=np.zeros((num_states, ))                  
    v_estimate= np.ones((num_states, ))
    old_v_estimate = np.ones((num_states, ))

    Niterations=10000
    if env.gamma > 0:
        epsilon = 0.0001*(1-env.gamma)/(2*env.gamma)
    else: 
        epsilon = 0.001

    for n_iter in range(Niterations):             
        for index in range(env.M*(env.B+1)):                                           
            v_estimate[index] = max([ Rew0[index] + env.gamma * np.dot(P0[index,:], old_v_estimate),  Rew1[index] + env.gamma * np.dot(P1[index,:], old_v_estimate)])   
        
        for index in range(env.M*(env.B+1) , num_states):                                
            v_estimate[index] = Rew1[index] + env.gamma *np.dot(P1[index,:], old_v_estimate)       
        diff = max(abs(v_estimate - old_v_estimate))
        
        if diff < epsilon:
            break
        

        for index in range(num_states):
            old_v_estimate[index] = v_estimate[index]


    for index in range(env.M *(env.B+1)):  
        if Rew0[index] + env.gamma * np.dot(P0[index,:], v_estimate) >= Rew1[index] + env.gamma * np.dot(P1[index,:], v_estimate):
            mu_[index] = 0
        else:
            mu_[index] = 1  
            
        mu_[index + env.M*(env.B+1)] = 1


    # show the solution

    mu_v0_=np.zeros((env.M,(env.B+1)))
    mu_v1_=np.zeros((env.M,(env.B+1)))
    for x in range(1, env.M+1):                                           
        for e in range(env.B+1):                                       
            index_0=compute_index(x,e,0, env.M, env.B)                   
            index_1=compute_index(x,e,1, env.M, env.B)                     
            mu_v0_[x-1,e]=mu_[index_0]             
            mu_v1_[x-1,e]=mu_[index_1]   
    
    if return_v_estimate:
        return mu_ , mu_v0_, old_v_estimate
    else:
        return mu_, mu_v0_


def compute_value_function(env, disadvantage_function, reward_function, cost_distribution, harvesting_distribution, policy):
    # value iteration algorithm, where the policy is fixed
    
    num_states = env.M*(env.B+1)*2
    cumulative_distribution = np.zeros(env.B+1+1, )
    for i in range(1, env.B+1+1):
        cumulative_distribution[i] = cumulative_distribution[i-1] + cost_distribution[i]

    # definition of the transition matrixes
    # transition probabilities when the action selected is a=0
    P0 = compute_P0(env, harvesting_distribution=harvesting_distribution)

    # Transition probabilities when a=1 "process"
    P1 = compute_P1(env, cost_distribution = cost_distribution, harvesting_distribution=harvesting_distribution)

    Rew1 = np.zeros((num_states, ))
    Rew0 = np.zeros((env.M*(env.B+1), ))

    beta = 1
    for x in range(1, env.M+1):
        for e in range(env.B+1):
            index=compute_index(x,e,0, env.M, env.B)
            # print(disadvantage_function)
            try: 
                if disadvantage_function.dtype == 'float64':

                    disadvantage = 0
                    fix_indexes = lambda x: -x*(x<0)
                    for h in range(1, len(harvesting_distribution)):
                        for c in range(1, len(cost_distribution)):
                            disadvantage += harvesting_distribution[h]*cost_distribution[c]*disadvantage_function[fix_indexes(e + h - c)]

                    Rew1[index] = R(x, 0, reward_function, env.M) - beta * disadvantage
            except:
                disadvantage = 0
                fix_indexes = lambda x: x*(x<0)
                for h in range(1, len(harvesting_distribution)):
                    for c in range(1, len(cost_distribution)):
                        disadvantage += harvesting_distribution[h]*cost_distribution[c]*disadvantage_function(fix_indexes(e + h - c))
                Rew1[index] = R(x, 0, reward_function, env.M) - beta * disadvantage
            
            Rew0[index] = R(x, 0, reward_function, env.M)
            index=compute_index(x,e,1, env.M, env.B)
            Rew1[index] = R(x,1, reward_function, env.M, env.delta)

    mu_=np.zeros((num_states, ))                  
    v_estimate= np.ones((num_states, ))
    old_v_estimate = np.ones((num_states, ))

    Niterations=10000
    epsilon = 0.0001*(1-env.gamma)/(2*env.gamma)

    for n_iter in range(Niterations):             
        for index in range(env.M*(env.B+1)):  
            # we do not consider the max, but rather just the result given by the fixed policy
            if policy[index] == 0:
                v_estimate[index] = Rew0[index] + env.gamma * np.dot(P0[index,:], old_v_estimate)
            elif policy[index] ==1:
                v_estimate[index] = Rew1[index] + env.gamma * np.dot(P1[index,:], old_v_estimate)
          
        
        for index in range(env.M*(env.B+1) , num_states):                                
            v_estimate[index] = Rew1[index] + env.gamma *np.dot(P1[index,:], old_v_estimate)       
        diff = max(abs(v_estimate - old_v_estimate))
        
        if diff < epsilon:
            break
        

        for index in range(num_states):
            old_v_estimate[index] = v_estimate[index]


    return v_estimate




def compute_stationary_distribution(stationary_transition_matrix):
    # stationary_distribution = np.zeros(len(recurrent_class))
    evals, evecs = np.linalg.eig(stationary_transition_matrix.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary_distribution = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary_distribution = stationary_distribution.real
    return stationary_distribution


def compute_transition_matrix_markov_chain(env, optimal_policy_vector):
    transition_matrix_markov_chain = np.zeros((env.num_states, env.num_states))
    # transition matrix of action a=0
    P0 = compute_P0(env, env.p_H)
    # transition matrix of action a=1
    P1 = compute_P1(env, env.p_C, env.p_H)
    for state in range(env.num_states):    
        action = optimal_policy_vector[state]
        if action == 0 and state < P0.shape[0]:
            for final_state in range(env.num_states):     
                transition_matrix_markov_chain[state, final_state] = P0[state, final_state]
        elif action == 1:
            for final_state in range(env.num_states):     
                transition_matrix_markov_chain[state, final_state] = P1[state, final_state]
    return transition_matrix_markov_chain


def length_path(env, optimal_policy, x, e, z):
    if z ==1:
        return x    # the action has been executed after x steps (length of the path equal to x)
    else:
        if optimal_policy[x-1, e] == 0:
            # action from current state is 0
            average_length_path_action_0 = 0
            for h in range(1, len(env.p_H)):
                average_length_path_action_0 += env.p_H[h] * length_path(env, optimal_policy, min(x+1, env.M), min(env.B, e+h), 0)
            return env.p_z * length_path(env, optimal_policy, min(x+1, env.M), e, 1) + (1 - env.p_z) * average_length_path_action_0
        else:
            # action = 1
            return x 

def exact_average_length(env, optimal_policy_vector, optimal_policy):
    list_length = []
    for e in range(env.B+1):
        length_server_1 = length_path(env, optimal_policy, 1, e, 1)
        length_server_0 = length_path(env, optimal_policy, 1, e, 0)
        list_length.append(env.p_z * length_server_1 + (1 - env.p_z)*length_server_0)
    return np.mean(list_length)

def exact_discounted_reward_function(env, optimal_policy_vector, optimal_policy):
    transition_matrix_markov_chain = compute_transition_matrix_markov_chain(env, optimal_policy_vector)    
    reward_vector = np.zeros((env.num_states, 1))
    for index in range(env.num_states):
        x, e, server_available = env.compute_state_values(index)

        if optimal_policy_vector[index] == 0:
            reward_vector[index] = R(x, 0, env.reward_function, env.M)
        else:
            if server_available:
                # server available
                reward_vector[index] = R(x,1, env.reward_function, env.M, env.delta)
            else:
                disadvantage = 0
                for h in range(1, len(env.p_H)):
                    for c in range(1, len(env.p_C)):
                        disadvantage += env.p_H[h]*env.p_C[c]*env.disadvantage_function(e + h - c)
                reward_vector[index] = R(x, 0, env.reward_function, env.M) - disadvantage
    
    matrix_to_invert = np.identity(env.num_states) - env.gamma * transition_matrix_markov_chain
    inverse_matrix = np.linalg.inv(matrix_to_invert)

    discounted_reward = np.matmul(inverse_matrix, reward_vector)
    
    # definition of the initial distirbution
    initial_distribution = np.zeros((1, env.num_states))
    for e in range(env.B+1):
        for server_available in range(2):
            index = env.compute_index(1, e, server_available)
            initial_distribution[0, index] = (server_available * env.p_z + (1 - server_available)*(1-env.p_z))/(env.B+1)
    initial_distribution = initial_distribution/np.sum(initial_distribution)

    discounted_reward = np.matmul(initial_distribution, discounted_reward)

    return float(discounted_reward) 