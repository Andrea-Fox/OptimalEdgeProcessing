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

def simultaneous_perturbation_stochastic_approximation(environments_list, n_devices, approximated_computation = False, perturbation_variance = 0.05, initial_p_z = None, initial_total_reward = None, return_list_rewards = False, initial_policy_list=None, initial_policy_vector_list=None, single_core = False):
    weight_function_ascent = (lambda x: 1/((x+2)**2))
    weight_function_perturbation = (lambda x: 5/(x+1)**2)

    perturbation_on_same_policy = True
    max_steps_from_last_update = 3


    if single_core:
        pool = None
    else:
        pool = mp.Pool(min(mp.cpu_count() -1, n_devices))

    if return_list_rewards:
        list_rewards = []
 
    if isinstance(initial_p_z, type(None)):
        p_z_list = np.zeros((n_devices))
        old_p_z_list = np.zeros((n_devices, ))
        for i in range(n_devices):
            p_z_list[i] = random.random()
            old_p_z_list[i] = p_z_list[i]
        optimal_total_reward = -math.inf
    else:
        p_z_list = initial_p_z
        old_p_z_list = initial_p_z
    p_z_list = probability_distr_projection(p_z_list)


    if isinstance(initial_total_reward, type(None)):
        optimal_total_reward = np.sum(compute_total_return(environments_list = environments_list, p_z_list = optimal_p_z, n_devices = n_devices, pool = pool, approximated_computation = approximated_computation))
    else:
        optimal_total_reward = initial_total_reward
        optimal_policy_vector_list = initial_policy_vector_list
        optimal_policy_list = initial_policy_list
    if not single_core:
        print(optimal_p_z, optimal_total_reward)
    if return_list_rewards:
        list_rewards.append(optimal_total_reward)
    

    verbose = False
    eps = 0.01
    done = False
    SPSA_steps = 1
    perturbation_variance = 0.05
    steps_from_last_update = 0


    while not done:
        if verbose:
            print('polling distribution = ', p_z_list)
        perturbation = np.random.normal(0, perturbation_variance, n_devices)
        p_z_positive_perturbation = p_z_list + weight_function_perturbation(SPSA_steps) * perturbation
        p_z_negative_perturbation = p_z_list - weight_function_perturbation(SPSA_steps) * perturbation
        p_z_positive_perturbation = probability_distr_projection(p_z_positive_perturbation)
        p_z_negative_perturbation = probability_distr_projection(p_z_negative_perturbation)
        if verbose:
            print('perturbation = ' + str(weight_function_perturbation(SPSA_steps) * perturbation))
            print('positive perturbation projection = ' + str(p_z_positive_perturbation))
            print('negative perturbation projection = ' + str(p_z_negative_perturbation))

        # computation of g: first we need to compute the value of the average reward for the new perturbated vectors
        if perturbation_on_same_policy:
            positive_perturbation_average_reward = compute_total_return(environments_list=environments_list, approximated_computation=approximated_computation, p_z_list=p_z_positive_perturbation, n_devices=n_devices, pool = pool, random_seed=SPSA_steps, optimal_policy_list=optimal_policy_list, optimal_policy_vector_list = optimal_policy_vector_list)
            negative_perturbation_average_reward = compute_total_return(environments_list=environments_list, approximated_computation=approximated_computation, p_z_list=p_z_negative_perturbation, n_devices=n_devices, pool = pool, random_seed=SPSA_steps, optimal_policy_list=optimal_policy_list, optimal_policy_vector_list = optimal_policy_vector_list)

        gradient_estimate = np.zeros((n_devices, )) 
        for i in range(n_devices):
            gradient_estimate[i] = (positive_perturbation_average_reward[i] - negative_perturbation_average_reward[i])/(2 * weight_function_perturbation(SPSA_steps)*perturbation[i])

        gradient_estimate = gradient_normalization(gradient_estimate)
        weighted_gradient_estimate = weight_function_ascent(SPSA_steps) * gradient_estimate
        if verbose:
            print('gradient estimate = ' + str(gradient_estimate))
            print('weighted gradient estimate = ' + str(weighted_gradient_estimate))

        # update parameters of the server polling distribution
        for i in range(n_devices):
            # try:
            p_z_list[i] = min(1, p_z_list[i] + weighted_gradient_estimate[i]) # weight_function_ascent(SPSA_steps) * gradient_estimate[i])
            # except:
            #     p_z_list[i] = 1

        p_z_list = probability_distr_projection(p_z_list)

        # now we first find the new optimal policy and then compute the total return of the system
        optimal_policy_list, optimal_policy_vector_list = find_approximated_optimal_policies(environments_list=environments_list, approximated_computation=True, p_z_list=p_z_list, n_devices=n_devices, single_core = True)

        discounted_reward_new_distribution = np.sum(compute_total_return(environments_list=environments_list, approximated_computation=approximated_computation, p_z_list=p_z_list, n_devices=n_devices, pool = pool, optimal_policy_list=optimal_policy_list, optimal_policy_vector_list=optimal_policy_vector_list))
        if return_list_rewards:
            list_rewards.append(discounted_reward_new_distribution)
        
        if discounted_reward_new_distribution > optimal_total_reward:
            # we have improved and need to update the distribution
            SPSA_steps += 1
            if verbose:
                print(SPSA_steps, p_z_list, discounted_reward_new_distribution, ' Updated values')
                print(old_p_z_list, p_z_list)
                print(np.sum((old_p_z_list-p_z_list)**2))
            # if the update is small, end the process
            if math.sqrt(np.sum((weighted_gradient_estimate)**2)) < eps:
                done = True 
            for i in range(n_devices):
                old_p_z_list[i] = p_z_list[i]
            optimal_total_reward = discounted_reward_new_distribution
            steps_from_last_update = 0
        else:
            if verbose:
                print(SPSA_steps, p_z_list, discounted_reward_new_distribution)
            for i in range(n_devices):
                p_z_list[i] = old_p_z_list[i]

        steps_from_last_update += 1
        if steps_from_last_update >= max_steps_from_last_update:
            done = True
        if verbose:
            print('------------------------------------------')

    if return_list_rewards:
            return p_z_list, optimal_total_reward, list_rewards
    else:
        return p_z_list, optimal_total_reward, None


def naive_random_search_method(environments_list, n_devices, approximated_computation, initial_p_z = None, initial_total_reward = None, return_list_rewards = False, single_core = False, fixed_number_attempts = None):
    
    if single_core:
        pool = None
    else:
        pool = mp.Pool(min(mp.cpu_count() -1, n_devices))

    k = 0
    bias_vector = np.zeros((n_devices, )) 
    deviation_variance = 0.05
    done = False
    steps_from_last_update = 0
    max_steps_from_last_update = 3
    total_attempts = 0
    verbose = False
    if return_list_rewards:
        list_rewards = []

    # find initial polling distribution and compute initial total reward
    if isinstance(initial_p_z, type(None)):
        optimal_p_z = np.zeros((n_devices))
        for i in range(n_devices):
            optimal_p_z[i] = random.random()
        optimal_total_reward = -math.inf
    else:
        optimal_p_z = initial_p_z
    optimal_p_z = probability_distr_projection(optimal_p_z)


    if isinstance(initial_total_reward, type(None)):
        optimal_total_reward = np.sum(compute_total_return(environments_list = environments_list, p_z_list = optimal_p_z, n_devices = n_devices, pool = pool, approximated_computation = approximated_computation))
    else:
        optimal_total_reward = initial_total_reward
    if not single_core:
        print(optimal_p_z, optimal_total_reward)
    if return_list_rewards:
        list_rewards.append(optimal_total_reward)

    
    while not done:
        # step 1
        new_p_z = np.ones((n_devices))
        while np.sum(new_p_z) > 1:
            new_p_z = np.zeros((n_devices))
            for i in range(n_devices):
                new_p_z[i] = random.random()
                
        if not single_core:
            print(new_p_z)
        total_return_new_parameter = np.sum(compute_total_return(environments_list = environments_list, p_z_list = new_p_z, n_devices = n_devices, pool = pool, approximated_computation = approximated_computation))
        if return_list_rewards:
            list_rewards.append(total_return_new_parameter)

        
        if total_return_new_parameter > optimal_total_reward:
            if verbose:
                print(str(total_return_new_parameter) + ' Updated values')
            steps_from_last_update = 0
            optimal_total_reward = total_return_new_parameter
            optimal_p_z = new_p_z
        else:
            if verbose:
                print(str(total_return_new_parameter))
            
        steps_from_last_update += 1
        total_attempts +=1
        if not isinstance(fixed_number_attempts, type(None)):
            if total_attempts >= total_attempts:
                done = True
        elif steps_from_last_update >= max_steps_from_last_update:
            done = True 
    if return_list_rewards:
            return optimal_p_z, optimal_total_reward, list_rewards
    else:
        return optimal_p_z, optimal_total_reward, None


def enhanced_localized_random_search(environments_list, n_devices, approximated_computation, initial_p_z = None, initial_total_reward = None, return_list_rewards = False, single_core = False):
    if single_core:
        pool = None
    else:
        pool = mp.Pool(min(mp.cpu_count() -1, n_devices))

    k = 0
    bias_vector = np.zeros((n_devices, )) 
    deviation_variance = 0.1
    done = False
    steps_from_last_update = 0
    max_steps_from_last_update = 3
    verbose = False
    if return_list_rewards:
        list_rewards = []

    # find initial polling distribution and compute initial total reward
    if isinstance(initial_p_z, type(None)):
        optimal_p_z = np.zeros((n_devices))
        for i in range(n_devices):
            optimal_p_z[i] = random.random()
        optimal_total_reward = -math.inf
    else:
        optimal_p_z = initial_p_z
    optimal_p_z = probability_distr_projection(optimal_p_z)


    if isinstance(initial_total_reward, type(None)):
        optimal_total_reward = np.sum(compute_total_return(environments_list = environments_list, p_z_list = optimal_p_z, n_devices = n_devices, pool = pool, approximated_computation = approximated_computation))
    else:
        optimal_total_reward = initial_total_reward
    if not single_core:
        print(optimal_p_z, optimal_total_reward)
    if return_list_rewards:
        list_rewards.append(optimal_total_reward)

    
    while not done:
        # step 1
        optimal_p_z = probability_distr_projection(optimal_p_z)
        if not single_core:
            print('Current p_z = ', optimal_p_z)

        deviation_vector =  np.random.normal(0, deviation_variance, n_devices)

        new_p_z = optimal_p_z + deviation_vector + bias_vector
        new_p_z = probability_distr_projection(new_p_z)

        total_return_new_parameter = np.sum(compute_total_return(environments_list = environments_list, p_z_list = new_p_z, n_devices = n_devices, pool = pool, approximated_computation = approximated_computation))
        if return_list_rewards:
            list_rewards.append(total_return_new_parameter)

        
        if total_return_new_parameter > optimal_total_reward:
            if verbose:
                print(str(total_return_new_parameter) + ' Updated values')
            steps_from_last_update = 0
            optimal_total_reward = total_return_new_parameter
            optimal_p_z = new_p_z
            bias_vector = 0.2 * bias_vector + 0.4 * deviation_vector
        else:
            if verbose:
                print(str(total_return_new_parameter))
            new_p_z = optimal_p_z + bias_vector - deviation_vector
            new_p_z = probability_distr_projection(new_p_z)
            total_return_new_parameter = np.sum(compute_total_return(environments_list = environments_list, p_z_list = new_p_z, n_devices = n_devices, pool = pool, approximated_computation = approximated_computation))
            if return_list_rewards:
                list_rewards.append(total_return_new_parameter)
            
            if total_return_new_parameter > optimal_total_reward:
                if verbose:
                    print(str(total_return_new_parameter) + 'Updated values')
                steps_from_last_update = 0
                optimal_total_reward = total_return_new_parameter
                optimal_p_z = new_p_z
                bias_vector = bias_vector - 0.4 * deviation_vector
            else:
                bias_vector = 0.5 * bias_vector
                if verbose:
                    print(str(total_return_new_parameter))
        steps_from_last_update += 1
        if steps_from_last_update >= max_steps_from_last_update:
            done = True 
    if return_list_rewards:
            return optimal_p_z, optimal_total_reward, list_rewards
    else:
        return optimal_p_z, optimal_total_reward, None


