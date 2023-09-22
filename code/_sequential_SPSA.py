from _mdp import BatteryBuffer
from _learning_methods import *
from _value_iteration_helper import *
from _stochastic_approximation_helper import *

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'


import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp


def SPSA_fixed_policy(environments_list, approximated_computation, initial_p_z, initial_total_reward, fixed_policy_vector, fixed_policy, n_devices, pool = None, first_attempt = False):
    weight_function_ascent_optimized = (lambda x: 1/((x+1)**2))     # exponent was equal to 3
    weight_function_perturbation_optimized = (lambda x: 5/(x+1)**2)

    # we compute the optimal policy for each device given the device
    optimal_policy_vector_list = []
    optimal_policy_list = []
    env_list = []

    # we perform SPSA, while maintaining the same policy for each device 
    # (and using only policy evaluation for computing the )
    eps = 0.01
    done = False
    SPSA_steps = 1

    if first_attempt:
        max_updates = 10
    else:
        max_updates = 10

    perturbation_variance = 0.05
    steps_from_last_update = 0

    
    optimal_p_z_list = np.zeros((n_devices, ))
    p_z_list = np.zeros((n_devices, ))
    for i in range(len(optimal_p_z_list)):
        optimal_p_z_list[i] = initial_p_z[i]
        p_z_list[i] = initial_p_z[i]
    optimal_total_reward = initial_total_reward
    while not done:
        perturbation = []
        for _ in range(n_devices):
            perturbation.append(random.gauss(0, perturbation_variance))
        
        perturbation = perturbation * np.ones((n_devices, ))
        p_z_positive_perturbation = optimal_p_z_list + weight_function_perturbation_optimized(SPSA_steps) * perturbation
        p_z_negative_perturbation = optimal_p_z_list - weight_function_perturbation_optimized(SPSA_steps) * perturbation
        p_z_positive_perturbation = probability_distr_projection(p_z_positive_perturbation)
        p_z_negative_perturbation = probability_distr_projection(p_z_negative_perturbation)

        # computation of gradient: first we need to compute the value of the average reward for the new perturbated vectors
        positive_perturbation_average_reward = compute_total_return(environments_list = environments_list, approximated_computation= approximated_computation, p_z_list = p_z_positive_perturbation, random_seed = SPSA_steps, pool = pool, n_devices = n_devices, optimal_policy_vector_list = fixed_policy_vector, optimal_policy_list = fixed_policy)
        negative_perturbation_average_reward = compute_total_return(environments_list = environments_list, approximated_computation= approximated_computation, p_z_list = p_z_negative_perturbation, random_seed = SPSA_steps, pool = pool, n_devices = n_devices, optimal_policy_vector_list = fixed_policy_vector, optimal_policy_list = fixed_policy)
        

        gradient_estimate = np.zeros((n_devices, ))
        for i in range(n_devices):
            gradient_estimate[i] = (positive_perturbation_average_reward[i] - negative_perturbation_average_reward[i])/(2 * weight_function_perturbation_optimized(SPSA_steps)*perturbation[i])
            # gradient_estimate[i] = (positive_perturbation_average_reward[i]*negative_perturbation_average_length[i] - positive_perturbation_average_length[i]* negative_perturbation_average_reward[i])/(2 * weight_function_perturbation(SPSA_steps)*perturbation[i])
        gradient_estimate = gradient_normalization(gradient_estimate)
        # update parameters of the server polling distribution
        for i in range(n_devices):
            min_perturbated_coordinate_i = 0 #  min(p_z_positive_perturbation[i], p_z_negative_perturbation[i])
            max_perturbated_coordinate_i = 1 # max(p_z_positive_perturbation[i], p_z_negative_perturbation[i])
            try:
                p_z_list[i] = min(max_perturbated_coordinate_i, max(optimal_p_z_list[i] + weight_function_ascent_optimized(SPSA_steps) * gradient_estimate[i] , min_perturbated_coordinate_i))
            except:
                p_z_list[i] = 1

        p_z_list = probability_distr_projection(p_z_list)
        # check if we have reached convergence
        total_reward_new_parameter = np.sum(compute_total_return(environments_list = environments_list, approximated_computation = approximated_computation, p_z_list = p_z_list, optimal_policy_vector_list = fixed_policy_vector, optimal_policy_list = fixed_policy, pool = pool, n_devices = n_devices))
        #  total_reward_new_parameter = np.sum(device_reward_new_distribution)
        if total_reward_new_parameter > optimal_total_reward:
            if not isinstance(pool, type(None)):
                print('Update attempt: ' + str(p_z_list) +' '+ str(total_reward_new_parameter)+'\tImprovement found')
            # we have improved and need to update the distribution
            SPSA_steps += 1
            # if the update is small, end the process
            # print(np.sum((optimal_p_z_list-p_z_list)**2))
            if np.sum((optimal_p_z_list-p_z_list)**2) < eps:
                done = True 
            for i in range(n_devices):
                optimal_p_z_list[i] = p_z_list[i]
            optimal_total_reward = total_reward_new_parameter
            steps_from_last_update = 0
            # print(optimal_p_z_list)
        else:
            if not isinstance(pool, type(None)):
                print('Update attempt: ' + str(p_z_list), str(total_reward_new_parameter))


        steps_from_last_update += 1
        if steps_from_last_update >= max_updates:
            done = True
        # print('------------------------------------------')
    
    return optimal_p_z_list, optimal_total_reward, (SPSA_steps > 1)


def sequential_SPSA(n_devices, k = 1, environments_list = None, approximated_computation = False, initial_p_z = None, return_list_rewards = False, initial_total_reward = None, initial_policy_list = None, initial_policy_vector_list = None, single_core  = None):
    
    disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]

    if isinstance(environments_list, type(None)):
        ValueError('Missing environments list')
    if n_devices <= 0:
        ValueError('Value of n_devices non positive')
  
    if not single_core:
        pool = mp.Pool(min(mp.cpu_count()-1 , n_devices))
    else:
        pool = None

    # find the initial (random) vector by finding k vectors, computing their corresponding total reward and finally
    # doing the optimization on the best initial value
    best_total_reward = -math.inf
    list_vectors = []
    if return_list_rewards:
        list_return = []

    if isinstance(initial_p_z, type(None)): 
        for initial_index in range(k):
            # random.seed(initial_index)
            initial_p_z = np.zeros((n_devices, ))
            for i in range(n_devices):
                initial_p_z[i] = random.random()
            initial_p_z = probability_distr_projection(initial_p_z)

            list_vectors.append(initial_p_z)
    else:
        list_vectors.append(initial_p_z)
    

    if isinstance(initial_total_reward, type(None)):
        for initial_index in range(k):
            # random.seed(initial_index)
            initial_p_z = list_vectors[initial_index]


            # for each vector and each device, compute the optimal policy and the reward (long path reward)
            # for brevity reasons, we first compute only the total reward of all the different values. Then 
            # we select the best vectors and compute the corresponding optimal policies
            initial_policy_vector_list = []
            initial_policy_list = []

            if not approximated_computation:
                for device_index in range(n_devices):
                    env = environments_list[device_index]
                    env.p_z = initial_p_z[device_index]    
                    env.disadvantage_function = disadvantage_function_list[device_index]
                    initial_policy_vector, initial_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H, random_seed=device_index)
                    initial_policy_vector_list.append(initial_policy_vector)
                    initial_policy_list.append(initial_policy)
            else:
                for device_index in range(n_devices):
                    environments_list[device_index].p_z = initial_p_z[device_index]    
                    environments_list[device_index].disadvantage_function = disadvantage_function_list[device_index]
                    # now we should find the approximated optimal policy
                initial_policy_list, initial_policy_vector_list = find_approximated_optimal_policies(environments_list=environments_list, approximated_computation=True, p_z_list = initial_p_z, n_devices=n_devices, pool = pool)

            initial_total_reward = np.sum(compute_total_return(environments_list = environments_list, approximated_computation = approximated_computation, p_z_list = initial_p_z, optimal_policy_vector_list = initial_policy_vector_list, optimal_policy_list = initial_policy_list, pool = pool, n_devices = n_devices))
            if return_list_rewards:
                list_return.append(initial_total_reward)
            if not single_core:
                print('Initial values, attempt ' + str(initial_index+1) + ':')
                print(initial_p_z, initial_total_reward)

            if initial_total_reward > best_total_reward:
                best_total_reward = initial_total_reward
                best_initial_p_z = initial_p_z
                best_initial_policy_list = initial_policy_list
                best_initial_policy_vector_list = initial_policy_vector_list
    else:
        # we have already one initial distribution and its total reward
        if return_list_rewards:
            list_return.append(initial_total_reward)
        best_total_reward = initial_total_reward
        best_initial_p_z = initial_p_z
        best_initial_policy_vector_list = initial_policy_vector_list
        best_initial_policy_list = initial_policy_list

    
    # perturbate the initial vector until we reach optimality
    if not single_core:
        print('Best initial p_Z = ' + str(best_initial_p_z))

    done = False

    p_z_distribution = best_initial_p_z
    optimal_reward = best_total_reward
    fixed_policy_vector = best_initial_policy_vector_list
    fixed_policy = best_initial_policy_list
    previous_policy_unchanged = False
    first_attempt = True
    while not done:

        # we apply SPSA algorithm, with the difference that we compute the optimal policy only once 
        # (in a future more optimized version, we exploit the optimal policy already computed when 
        # computing the total reward)
        optimal_p_z_distribution, total_reward_fixed_policies, improvement_found = SPSA_fixed_policy(environments_list = environments_list,initial_p_z = p_z_distribution, initial_total_reward = optimal_reward, fixed_policy_vector = fixed_policy_vector, fixed_policy = fixed_policy, n_devices=n_devices, pool = pool, approximated_computation = approximated_computation, first_attempt=first_attempt)
        first_attempt = False

        if improvement_found:
            if not single_core:
                print('Optimization completed on the policies given by ' + str(p_z_distribution) + ':')
                print(optimal_p_z_distribution, total_reward_fixed_policies)

            # the value printed above is the optimal reward considering the evaluation done on the 
            # curves given by the initial values of the parameters

            # we now find, for each device, the optimal policy of the optimal parameter and the consequent
            # (higher) total reward of the system
            optimal_policy_vector_list = []
            optimal_policy_list = []

            if not approximated_computation:
                for device_index in range(n_devices):
                    env = environments_list[device_index]
                    env.p_z = optimal_p_z_distribution[device_index]
                    if not isinstance(env.disadvantage_function_exponent, type(None)):
                        env.disadvantage_function = (lambda e: (e<0) *(-e)**(env.disadvantage_function_exponent))
                    else:
                        env.disadvantage_function = disadvantage_function_list[device_index]
                    
                    optimized_distr_policy_vector, optimized_distr_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H, random_seed=device_index)
                    optimal_policy_vector_list.append(optimized_distr_policy_vector)
                    optimal_policy_list.append(optimized_distr_policy)
            else:
                for device_index in range(n_devices):
                    environments_list[device_index].p_z = optimal_p_z_distribution[device_index] 
                    if not isinstance(environments_list[device_index].disadvantage_function_exponent, type(None)):
                        environments_list[device_index].disadvantage_function = (lambda e: (e<0) *(-e)**(environments_list[device_index].disadvantage_function_exponent))
                    else:
                        environments_list[device_index].disadvantage_function = disadvantage_function_list[device_index]
       
                    environments_list[device_index].disadvantage_function = disadvantage_function_list[device_index]
                # now we should find the approximated optimal policy
                optimal_policy_list, optimal_policy_vector_list = find_approximated_optimal_policies(environments_list=environments_list, approximated_computation=True, p_z_list = optimal_p_z_distribution, n_devices=n_devices, pool = pool)


            optimal_total_reward = np.sum(compute_total_return(environments_list = environments_list, approximated_computation = approximated_computation, p_z_list =optimal_p_z_distribution, optimal_policy_vector_list = optimal_policy_vector_list, optimal_policy_list = optimal_policy_list, pool = pool, n_devices = n_devices))
            
            if approximated_computation:
                min_improvement = 0.01
            else:
                min_improvement = 0.1

            # if the total reward with the appropriate policy is lower than wht was found before, we keep the same policy as before and restart the SPSA 
            if optimal_total_reward > total_reward_fixed_policies:
                if not single_core:
                    print('Total reward with the appropriate policies: ', optimal_p_z_distribution, optimal_total_reward)
                if return_list_rewards:
                    list_return.append(optimal_total_reward)
                if math.fabs(optimal_total_reward - total_reward_fixed_policies)**2 < min_improvement:
                    done = True
                previous_policy_unchanged = False
                p_z_distribution = optimal_p_z_distribution
                optimal_reward = optimal_total_reward
                fixed_policy_vector = optimal_policy_vector_list
                fixed_policy = optimal_policy_list
            else:
                if return_list_rewards:
                    list_return.append(total_reward_fixed_policies)
                if not single_core:
                    print('The optimal policy remains unchnaged')
                p_z_distribution = optimal_p_z_distribution
                if previous_policy_unchanged:
                    done = True
                previous_policy_unchanged = True
                optimal_reward = total_reward_fixed_policies
                optimal_total_reward = total_reward_fixed_policies
                # fixed_policy_vector = optimized_distr_policy_vector
                # fixed_policy = optimized_distr_policy\
            if not single_core:
                print('----------------------------------------')

        else:
            done = True
            optimal_total_reward = optimal_reward
        

        
    if return_list_rewards:
            return optimal_p_z_distribution, optimal_total_reward, list_return
    else:
        return optimal_p_z_distribution, optimal_total_reward, None