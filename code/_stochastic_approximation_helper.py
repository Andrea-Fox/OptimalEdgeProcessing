from _mdp import BatteryBuffer
from _learning_methods import *
from _value_iteration_helper import *

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

def function_to_maximise(x):
    return -(x-1/3)**2 +1

def compute_epsilon(n_devices):
    return 0.01


def gradient_normalization(gradient):
    # the sum of absolute values will become one. Sings will remain unchanged
    absolute_value_gradient = np.zeros((gradient.shape))
    for i in range(gradient.shape[0]):
        absolute_value_gradient[i] = math.fabs(gradient[i])
    absolute_value_gradient = probability_distr_projection(absolute_value_gradient)
    
    for i in range(gradient.shape[0]):
        if gradient[i] < 0:
            gradient[i] = -absolute_value_gradient[i]
        else:
            gradient[i] = absolute_value_gradient[i]

    return gradient 

def probability_distr_projection(distr):
    # for the indexes corresponing to a value above 0, we consider the normalized vector. Elements below 0 will become equal to 0
    elements_to_consider = np.zeros((len(distr), ))
    max_value = 0
    for index in range(len(distr)):
        if distr[index]<0:
            distr[index] = 0
    if np.sum(distr) > 1:
        return distr/np.sum(distr)
    return distr


def parallelization_handler_reward(env, approximated_computation, random_seed, device_index, n_devices, p_z, optimal_policy_vector = None, optimal_policy = None):

    current_average_rewards = np.zeros((n_devices, ))
    current_average_lengths = np.zeros((n_devices, ))
    disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]

    # env = BatteryBuffer(M = M_list[device_index], B = B_list[device_index], h=h_list[device_index], p_z=p_z_list[device_index], cost_probability=cost_probability_list[device_index], reward_function=reward_function_list[device_index], disadvantage_function = disadvantage_function_list[device_index], gamma = gamma_list[device_index], eps_0 = eps0_list[device_index], exploration_rate=exploration_rate_list[device_index]) 
    # env = environments_list[device_index]
    if not isinstance(env.disadvantage_function_exponent, type(None)):
        env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
    else:
        env.disadvantage_function = disadvantage_function_list[device_index]
    
    env.p_z = p_z # _list[device_index]

    
    if isinstance(optimal_policy_vector, type(None)) and isinstance(optimal_policy, type(None)):
        if approximated_computation:
            n_attempts = 5
            num_episodes = 3001
            discounted_rewards_list = []
            list_q_value_tables = []
            learning_rate = [.1, .25, .5, .5, .5]
            beta = [.51, .75, .75, .9, .75]
            n = [1, 1, 1, 1, 3]    
            for attempt_index in range(n_attempts):
                q_values, discounted_reward, _ = stairway_q_learning(env = env, num_episodes = num_episodes, exploration_rate = 0.5, only_last_measurement=True, learning_rate= learning_rate[attempt_index], n = n[attempt_index], beta= beta[attempt_index])
                discounted_rewards_list.append(discounted_reward)
                list_q_value_tables.append(q_values)

            return max(discounted_rewards_list)
        else:
            optimal_policy_vector, optimal_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H, random_seed=random_seed)
            current_average_rewards[device_index] = exact_discounted_reward_function(env, optimal_policy_vector, optimal_policy)
            # print(p_z_list[device_index], average_rewards_list)
            return current_average_rewards[device_index]
    else:

        if approximated_computation:
            current_discounted_rewards = compute_average_reward(env, optimal_policy_vector) 
        else:
            current_discounted_rewards = exact_discounted_reward_function(env, optimal_policy_vector, optimal_policy)
        return current_discounted_rewards

def compute_total_return(environments_list, approximated_computation, p_z_list, n_devices, pool = None, random_seed = -1, optimal_policy_vector_list = None, optimal_policy_list = None):
    if isinstance(optimal_policy_vector_list, type(None)) and isinstance(optimal_policy_list, type(None)):
        # print('Search for the optimal policies for the vector ' +str(p_z_list))
        if isinstance(pool, type(None)):
            list_results = []
            disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]
            for device_index in range(n_devices):
                env = environments_list[device_index]
                if not isinstance(env.disadvantage_function_exponent, type(None)):
                    env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
                else:
                    env.disadvantage_function = disadvantage_function_list[device_index]
                env.p_z = p_z_list[device_index]
                if approximated_computation:
                    n_attempts = 5
                    num_episodes = 3001
                    discounted_rewards_list = []
                    list_q_value_tables = []
                    learning_rate = [.1, .25, .5, .5, .5]
                    beta = [.51, .75, .75, .9, .75]
                    n = [1, 1, 1, 1, 3]    
                    for attempt_index in range(n_attempts):
                        q_values, discounted_reward, _ = stairway_q_learning(env = env, num_episodes = num_episodes, exploration_rate = 0.5, only_last_measurement=True, learning_rate= learning_rate[attempt_index], n = n[attempt_index], beta= beta[attempt_index])
                        discounted_rewards_list.append(discounted_reward)
                        list_q_value_tables.append(q_values)
                    list_results.append(max(discounted_rewards_list))
                else:
                    optimal_policy_vector, optimal_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H, random_seed=random_seed)
                    current_average_rewards[device_index] = exact_discounted_reward_function(env, optimal_policy_vector, optimal_policy)
                    # print(p_z_list[device_index], average_rewards_list)
                    list_results.append(current_average_rewards[device_index]) 

        else:
            for i in range(n_devices):
                env = environments_list[i]
                env.disadvantage_function = 0
            list_results = pool.starmap(parallelization_handler_reward, [(environments_list[device_index], approximated_computation, random_seed, device_index, n_devices, p_z_list[device_index]) for device_index in range(n_devices)]  )
    else:
        if isinstance(pool, type(None)):
            list_results = []
            # we need to repeat the same operations as in parallelization_handler_reward
            disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]
            for device_index in range(n_devices):
                env = environments_list[device_index]
                
                if not isinstance(env.disadvantage_function_exponent, type(None)):
                    env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
                else:
                    env.disadvantage_function = disadvantage_function_list[device_index]
                
                env.p_z = p_z_list[device_index]
                if approximated_computation:
                    current_discounted_rewards = compute_average_reward(env, optimal_policy_vector_list[device_index]) 
                else:
                    current_discounted_rewards = exact_discounted_reward_function(env, optimal_policy_vector_list[device_index], optimal_policy_list[device_index])
                list_results.append(current_discounted_rewards)

        else:
            for i in range(n_devices):
                env = environments_list[i]
                env.disadvantage_function = 0
            list_results = pool.starmap(parallelization_handler_reward, [(environments_list[device_index], approximated_computation, random_seed, device_index, n_devices, p_z_list[device_index], optimal_policy_vector_list[device_index], optimal_policy_list[device_index]) for device_index in range(n_devices)]  )
    return list_results


def parallelization_handler_policy(environments_list, device_index, n_devices, p_z_list):
    print(device_index)
    current_average_rewards = np.zeros((n_devices, ))
    current_average_lengths = np.zeros((n_devices, ))
    disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]

    # env = BatteryBuffer(M = M_list[device_index], B = B_list[device_index], h=h_list[device_index], p_z=p_z_list[device_index], cost_probability=cost_probability_list[device_index], reward_function=reward_function_list[device_index], disadvantage_function = disadvantage_function_list[device_index], gamma = gamma_list[device_index], eps_0 = eps0_list[device_index], exploration_rate=exploration_rate_list[device_index]) 
    env = environments_list[device_index]
    if not isinstance(env.disadvantage_function_exponent, type(None)):
        env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
    else:
        env.disadvantage_function = disadvantage_function_list[device_index]
        env.p_z = p_z_list[device_index]

    # We now compute the optimal solution using Stairway Q-learning with standard parameters (we should do some parameters 
    # tuning to have better performances). We will perform the algorithm n_attempts (n_attempts = 1) times and then consider 
    # the policy which gives the highest value of the average reward. Each attempt will stop after 1000 episodes  

    # could be a good idea to consider different values of the hyperparameters and do one attempt for each set of these
    # the hyperparameters considered in the learning part were
    # learning_rate_list = [.05, .1, .15, .25, .5]
    # beta_list = [.51,  .75, .9, 1]
    # n_list = [1, 3, 5]
    # we clearly only consider a subset of all possible combinations
    n_attempts = 5
    num_episodes = 3001
    discounted_rewards_list = []
    list_q_value_tables = []
    learning_rate = [.1, .25, .5, .5, .5]
    beta = [.51, .75, .75, .9, .75]
    n = [1, 1, 1, 1, 3]    
    for attempt_index in range(n_attempts):
        q_values, discounted_reward, _ = stairway_q_learning(env = env, num_episodes = num_episodes, exploration_rate = 0.5, only_last_measurement=True, learning_rate= learning_rate[attempt_index], n = n[attempt_index], beta= beta[attempt_index])
        discounted_rewards_list.append(discounted_reward)
        list_q_value_tables.append(q_values)
    # print(device_index, p_z_list[device_index], average_rewards_list)
    return list_q_value_tables[np.argmax(discounted_rewards_list)]
     

def find_approximated_optimal_policies(environments_list, approximated_computation, p_z_list, n_devices, pool = None, random_seed = -1, optimal_policy_vector_list = None, optimal_policy_list = None, single_core = False):
    if isinstance(pool, type(None)) or single_core:
        # we need to write the code to work as in parallelization_handler_policy
        disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]
        list_results = []
        for device_index in range(n_devices):
            env = environments_list[device_index]
            if not isinstance(env.disadvantage_function_exponent, type(None)):
                env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
            else:
                env.disadvantage_function = disadvantage_function_list[device_index]
            env.p_z = p_z_list[device_index]

            n_attempts = 5
            num_episodes = 3001
            discounted_rewards_list = []
            list_q_value_tables = []
            learning_rate = [.1, .25, .5, .5, .5]
            beta = [.51, .75, .75, .9, .75]
            n = [1, 1, 1, 1, 3]    
            for attempt_index in range(n_attempts):
                q_values, discounted_reward, _ = stairway_q_learning(env = env, num_episodes = num_episodes, exploration_rate = 0.5, only_last_measurement=True, learning_rate= learning_rate[attempt_index], n = n[attempt_index], beta= beta[attempt_index])
                discounted_rewards_list.append(discounted_reward)
                list_q_value_tables.append(q_values)
            list_results.append(list_q_value_tables[np.argmax(discounted_rewards_list)]) 

    else:

        print('Search for the optimal policies for the vector ' +str(p_z_list))
        for i in range(n_devices):
            env = environments_list[i]
            env.disadvantage_function = 0
        list_results = pool.starmap(parallelization_handler_policy, [(environments_list, device_index, n_devices, p_z_list) for device_index in range(n_devices)] )
    # list_result is a list of q_values tables: we need ot translate it into a list of policies, with both a list of vectors and a list of matrixes
    
    list_policies = []
    list_policy_vectors = []
    for device_index in range(n_devices):
        env = environments_list[device_index]
        q_values = list_results[device_index]
        policy = np.zeros((env.M, env.B+1))
        policy_vector = np.ones((env.num_states, ))
        for x in range(env.M):
            for e in range(env.B+1):
                index = env.compute_index(x, e, 0)
                policy[x-1, e] = np.argmax(q_values[index, :])
                policy_vector[index] = np.argmax(q_values[index, :])
        list_policies.append(policy)
        list_policy_vectors.append(policy_vector)
        
    
    
    return list_policies, list_policy_vectors 