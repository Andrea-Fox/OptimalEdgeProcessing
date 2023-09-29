# this script deals with the multiserver model, with the goal of finding the optimal distribution
# of the server probability
# For the moment we assume to have access to the actual value of the optimal average reward (and not the one given by SQL)
# we use SPSA algorithm with common random numbers (same random seed for both perturbations)

# idea of the algorithm: two timescales, one for finding the optimal policy for each device given the server
# polling  distribution and the other one to update the server polling distribution



# toDo:
# - implement a stopping mechanism in the learning method 
# - save also the amount of policy evaluations (create histograms even for that)
# - understand the time taken by each policy learning and policy evaluation
# - parallelize computation of optimal solutions in context with exact computations
# - when possible (small number of devices), execute parallely the different methods 
#   (even considering simultaneously different initial distributions to improve performances )




from _mdp import BatteryBuffer
from _learning_methods import *
from _value_iteration_helper import *
from _sequential_SPSA import *
from _stochastic_approximation_algorithms import *

import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

import multiprocessing as mp

def optimal_polling_distribution(initial_p_z, n_devices, SPSA_algorithm, naive_random_search,enhanced_random_search, sequential_SPSA_algorithm, sequential_SPSA_algorithm_exact, run_index, random_environments = True):

    random.seed(run_index)

    initial_p_z = 2 * np.ones((n_devices, ))
    while np.sum(initial_p_z) >1:
        for i in range(n_devices):
            initial_p_z[i] = random.random()

    print(run_index, initial_p_z)
    environments_list = []

    for device_index in range(n_devices):
        if random_environments:
            random_M = random.randint(10, 26)
            random_B = random.randint(10, 26)     
            cost_probability = 3
            random_p_z = random.randint(1, 10)/100
            random_mean_cost = random.randint(3, math.floor(random_B/2))
            random_sigma_cost = random.randint(1, 5)
            reward_function = np.arange(random_M, 0, -1)**1
            random_disadvantage_function_exponent = random.choices(np.arange(1, 5), weights= [.1, .2, .4, .3] )[0]
            random_gamma = random.randint(90, 99)/100
            env = BatteryBuffer(M = random_M, B = random_B, h=1, p_z= random_p_z, cost_probability=cost_probability, mean_value_normal_distr=random_mean_cost, reward_function=reward_function, disadvantage_function_exponent = random_disadvantage_function_exponent, gamma = random_gamma, sigma_normal_distr= random_sigma_cost) 
            env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)

        else:
            M_list = [15, 15, 15, 16, 17, 13, 14, 17, 15, 15] * np.ones((max_n_devices, ))
            B_list = [12, 20, 15, 20, 15, 12, 20, 15, 20, 15] * np.ones((max_n_devices, ))
            gamma_list = [.95, .99, .95, .99, .975, .95, .75, .95, .99, .9] * np.ones((max_n_devices, ))
            eps0_list = np.repeat(0, max_n_devices)
            delta_list = [1, 1, 1, 1, 3, 1, 1, 1, 4, 0] *  np.ones((max_n_devices, ))
            lambda_function_exponent_list = [4, 2, 4, 2, 4, 4, 2, 4, 2, 4] * np.ones((max_n_devices, ))
            disadvantage_function_list = [(lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**5), (lambda e: (e<0) *(-e)**3), (lambda e: (e<0) *(-e)**4), (lambda e: (e<0) *(-e)**2)]
            reward_function_list = [np.arange(M_list[0], 0, -1)**1, np.arange(M_list[1], 0, -1)**1, np.arange(M_list[2], 0, -1)**1, np.arange(M_list[3], 0, -1)**1, np.arange(M_list[4], 0, -1)**1, np.arange(M_list[5], 0, -1)**1, np.arange(M_list[6], 0, -1)**1, np.arange(M_list[7], 0, -1)**1, np.arange(M_list[8], 0, -1)**1, np.arange(M_list[9], 0, -1)**1]
            h_list = np.repeat(1, max_n_devices)
            cost_probability_list = np.repeat(3, max_n_devices)
            mean_cost_list = [4, 2, 5, 2, 4, 3, 6, 3, 2, 4] * np.ones((max_n_devices, ))
            M_list = M_list[:n_devices]
            B_list = B_list[:n_devices]
            gamma_list = gamma_list[:n_devices]
            eps0_list = eps0_list[:n_devices]
            delta_list = delta_list[:n_devices]
            lambda_function_exponent_list = lambda_function_exponent_list[:n_devices]
            disadvantage_function_list = disadvantage_function_list[:n_devices]
            reward_function_list = reward_function_list[:n_devices]
            h_list = h_list[:n_devices]
            cost_probability_list = cost_probability_list[:n_devices]
            mean_cost_list = mean_cost_list[:n_devices]
            
            env = BatteryBuffer(M = M_list[device_index], B = B_list[device_index], h=1, p_z=0, cost_probability= cost_probability_list[device_index], mean_value_normal_distr= mean_cost_list[device_index], reward_function = reward_function_list[device_index], disadvantage_function = disadvantage_function_list[device_index], gamma = gamma_list[device_index]) 
        environments_list.append(env)


    rewards_naive_random_search = 0
    rewards_enhanced_random_search = 0
    rewards_sequential_SPSA = 0
    rewards_sequential_SPSA_exact = 0
    rewards_SPSA_algorithm = 0 
    time_naive_random_search = 0
    time_enhanced_random_search = 0
    time_sequential_SPSA = 0
    time_sequential_SPSA_exact = 0
    time_SPSA_algorithm = 0

    unique_initial_policy, unique_initial_policy_vector = find_approximated_optimal_policies(environments_list=environments_list, approximated_computation=True, p_z_list=initial_p_z, n_devices=n_devices, single_core = True)
    
    unique_initial_reward = compute_total_return(environments_list, True, p_z_list=initial_p_z, n_devices = n_devices, optimal_policy_list=unique_initial_policy, optimal_policy_vector_list=unique_initial_policy_vector)
    unique_initial_reward = np.sum(unique_initial_reward)
    # print(unique_initial_reward)

    # print(0, p_z_list, old_total_reward)
    # print('------------------------------------------')

    # baseline
    _, _, reward_naive_random = naive_random_search_method(environments_list, n_devices, approximated_computation=True, single_core=True, return_list_rewards=True, fixed_number_attempts=10)
    baseline = np.mean(reward_naive_random)
    # print(str(run_index) + ' baseline')

    # exact computations
    start_time = time.time()
    optimal_p_z, optimal_total_reward, rewards_sequential_SPSA_exact = sequential_SPSA(n_devices = n_devices, k = 1, environments_list = environments_list, approximated_computation = False, return_list_rewards=True, initial_p_z = initial_p_z, initial_total_reward = unique_initial_reward, initial_policy_list=unique_initial_policy, initial_policy_vector_list=unique_initial_policy_vector, single_core=True)
    time_sequential_SPSA_exact = (time.time() - start_time)
    # print(str(run_index) + ' exact')

    
    if SPSA_algorithm:
        start_time_SPSA = time.time()
        optimal_p_z, optimal_total_reward, rewards_SPSA_algorithm = simultaneous_perturbation_stochastic_approximation(environments_list = environments_list, n_devices = n_devices, approximated_computation = True, return_list_rewards=True, initial_p_z=initial_p_z, initial_total_reward = unique_initial_reward, initial_policy_list=unique_initial_policy, initial_policy_vector_list=unique_initial_policy_vector, single_core= True)
        time_SPSA_algorithm = (time.time() - start_time_SPSA)
        # print(rewards_SPSA_algorithm)
        # print(str(run_index) + ' SPSA')

    
    if naive_random_search:
        start_time_naive_random_search = time.time()
        optimal_p_z, optimal_total_reward, rewards_naive_random_search = naive_random_search_method(environments_list = environments_list, n_devices = n_devices, approximated_computation = True, return_list_rewards=True, initial_p_z = initial_p_z, initial_total_reward = unique_initial_reward, single_core = True) 
        time_naive_random_search = time.time() - start_time_naive_random_search
        # print(str(run_index) + ' naive')

        # print(rewards_naive_random_search)
    if enhanced_random_search:
        start_time_enhanced_random_search = time.time()
        optimal_p_z, optimal_total_reward, rewards_enhanced_random_search = enhanced_localized_random_search(environments_list = environments_list, n_devices = n_devices, approximated_computation = True, return_list_rewards=True, initial_p_z = initial_p_z, initial_total_reward = unique_initial_reward, single_core = True) 
        time_enhanced_random_search = time.time() - start_time_enhanced_random_search
        # print(str(run_index) + ' enhanced')

        # print(rewards_enhanced_random_search)
    if sequential_SPSA_algorithm:
        start_time_sequential_SPSA = time.time()
        optimal_p_z, optimal_total_reward, rewards_sequential_SPSA = sequential_SPSA(n_devices = n_devices, k = 1, environments_list = environments_list, approximated_computation = True, return_list_rewards=True, initial_p_z = initial_p_z, initial_total_reward = unique_initial_reward, initial_policy_list=unique_initial_policy, initial_policy_vector_list=unique_initial_policy_vector, single_core = True)
        time_sequential_SPSA = time.time() - start_time_sequential_SPSA
        # print(str(run_index) + ' sequential SPSA')

        # print(rewards_sequential_SPSA)
    # we need to add the baseline

    print(run_index, initial_p_z, baseline, np.max(rewards_sequential_SPSA_exact), np.max(rewards_naive_random_search), np.max(rewards_enhanced_random_search), np.max(rewards_SPSA_algorithm), np.max(rewards_sequential_SPSA))

    return [rewards_sequential_SPSA_exact, time_sequential_SPSA_exact, rewards_naive_random_search, time_naive_random_search, rewards_enhanced_random_search, time_enhanced_random_search, rewards_sequential_SPSA, time_sequential_SPSA, rewards_SPSA_algorithm, time_SPSA_algorithm, baseline]




def parallel_computing_handler(list_unique_initial_distributions, n_devices, n_runs,SPSA_algorithm, naive_random_search,enhanced_random_search, sequential_SPSA_algorithm, sequential_SPSA_algorithm_exact, random_environments):
    
    pool = mp.Pool(min(mp.cpu_count() -1, n_runs))
    list_rewards_naive_random_search = [] 
    list_time_naive_random_search = []
    list_rewards_enhanced_random_search = [] 
    list_time_enhanced_random_search = []
    list_rewards_sequential_SPSA = []
    list_time_sequential_SPSA = []
    list_rewards_sequential_SPSA_exact = [] 
    list_time_sequential_SPSA_exact = []
    list_rewards_SPSA_algorithm = []
    list_time_SPSA_algorithm = []
    list_baseline = []


    parallel_result = pool.starmap(optimal_polling_distribution, [([], n_devices, SPSA_algorithm, naive_random_search, enhanced_random_search, sequential_SPSA_algorithm, sequential_SPSA_algorithm_exact, run_index, random_environments) for run_index in range(n_runs)])
    
    for i in range(n_initial_distributions):
        attempt = parallel_result[i]
        list_rewards_sequential_SPSA_exact.append(attempt[0])
        list_time_sequential_SPSA_exact.append(attempt[1]) 
        list_rewards_naive_random_search.append(attempt[2]) 
        list_time_naive_random_search.append(attempt[3]) 
        list_rewards_enhanced_random_search.append(attempt[4]) 
        list_time_enhanced_random_search.append(attempt[5]) 
        list_rewards_sequential_SPSA.append(attempt[6]) 
        list_time_sequential_SPSA_exact.append(attempt[7]) 
        list_rewards_SPSA_algorithm.append(attempt[8])
        list_time_SPSA_algorithm.append(attempt[9])
        list_baseline.append(attempt[10])

    return list_rewards_naive_random_search, list_time_naive_random_search, list_rewards_enhanced_random_search, list_time_enhanced_random_search, list_rewards_sequential_SPSA, list_time_sequential_SPSA, list_rewards_sequential_SPSA_exact, list_time_sequential_SPSA_exact, list_rewards_SPSA_algorithm, list_time_SPSA_algorithm, list_baseline
  
n_devices = 3
n_initial_distributions = 50

initial_time_all = time.time()


list_unique_initial_distributions = []

while len(list_unique_initial_distributions) < n_initial_distributions:

    unique_initial_p_z = np.zeros((n_devices, ))
    for i in range(n_devices):
        unique_initial_p_z[i] = 10
    if np.sum(unique_initial_p_z) <= math.inf:
        list_unique_initial_distributions.append(unique_initial_p_z)

random_environments = True

SPSA_algorithm = True
naive_random_search = True
enhanced_random_search = True
sequential_SPSA_algorithm = True
sequential_SPSA_algorithm_exact = True

list_rewards_naive_random_search, list_time_naive_random_search, list_rewards_enhanced_random_search, list_time_enhanced_random_search, list_rewards_sequential_SPSA, list_time_sequential_SPSA, list_rewards_sequential_SPSA_exact, list_time_sequential_SPSA_exact, list_rewards_SPSA_algorithm, list_time_SPSA_algorithm, list_baseline = parallel_computing_handler(list_unique_initial_distributions = [], n_devices = n_devices, n_runs = n_initial_distributions, SPSA_algorithm = SPSA_algorithm, naive_random_search = naive_random_search, enhanced_random_search = enhanced_random_search, sequential_SPSA_algorithm = sequential_SPSA_algorithm, sequential_SPSA_algorithm_exact = sequential_SPSA_algorithm_exact, random_environments= random_environments)

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/stochastic_approximation_methods_comparison/fixed_N/'
if random_environments:
    folder += str(n_devices) + '_devices/random_environments/'
else:
    folder +=  str(n_devices) + '_devices/fixed_environments/'
if not os.path.exists(folder):
    os.makedirs(folder)

if naive_random_search:
    print(list_rewards_naive_random_search)
    pk.dump(list_rewards_naive_random_search, open(folder + 'list_rewards_naive_random_search.dat', 'wb'))
    pk.dump(list_time_naive_random_search, open(folder + 'list_time_naive_random_search.dat', 'wb'))
if enhanced_random_search:
    pk.dump(list_rewards_enhanced_random_search, open(folder + 'list_rewards_enhanced_random_search.dat', 'wb'))
    pk.dump(list_time_enhanced_random_search, open(folder + 'list_time_enhanced_random_search.dat', 'wb'))
if SPSA_algorithm:
    pk.dump(list_rewards_SPSA_algorithm, open(folder + 'list_rewards_SPSA_algorithm.dat', 'wb'))
    pk.dump(list_time_SPSA_algorithm, open(folder + 'list_time_SPSA_algorithm.dat', 'wb'))
if sequential_SPSA_algorithm:
    print(list_rewards_sequential_SPSA)
    print(list_time_sequential_SPSA)
    pk.dump(list_rewards_sequential_SPSA, open(folder + 'list_rewards_sequential_SPSA.dat', 'wb'))
    pk.dump(list_time_sequential_SPSA, open(folder + 'list_time_sequential_SPSA.dat', 'wb'))
if sequential_SPSA_algorithm_exact:
    print(list_rewards_sequential_SPSA_exact)
    pk.dump(list_rewards_sequential_SPSA_exact, open(folder + 'list_rewards_sequential_SPSA_exact.dat', 'wb'))
    pk.dump(list_time_sequential_SPSA_exact, open(folder + 'list_time_sequential_SPSA_exact.dat', 'wb'))

pk.dump(list_baseline, open(folder + 'list_baseline.dat', 'wb'))



final_time = time.time()


print("--- %s seconds ---" % (final_time - initial_time_all))

# 10696.206598043442 seconds

