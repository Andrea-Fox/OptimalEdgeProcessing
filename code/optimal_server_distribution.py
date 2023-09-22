# For several different values of n_devices, we compare the performances of sequential SPSA.
# In particular, we will perform N (N=15) attempts for each value of n_device
# 
# For each value of n_attempt, we perform 5 exact calculations to find the real_max_reward (sort of...)
# and then execute only sequential SPSA
# 
# 
# It could be interesting consider all the time different environments and test the performances in different 
# settings: to do so it should be necessary to compute every time the optimal total reward




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


def optimal_polling_distribution(initial_p_z, n_devices, run_index):
    random.seed(run_index)

    environments_list = []

    # randomly define the parameters of the servers involved
    for device_index in range(n_devices):
        random_M = random.randint(10, 21)
        random_B = random.randint(10, 21)     # if M is big, B has to be a bit smaller to avoid too long calculations
        cost_probability = 3
        mean_cost = random.randint(3, math.floor(random_B/2))
        reward_function = np.arange(random_M, 0, -1)**1
        disadvantage_function_exponent = random.choices(np.arange(1, 5), weights= [.1, .2, .4, .3] )[0]
        gamma = random.randint(90, 99)/100
        
        env = BatteryBuffer(M = random_M, B = random_B, h=1, p_z=0, cost_probability=cost_probability, mean_value_normal_distr=mean_cost, reward_function=reward_function, disadvantage_function_exponent= disadvantage_function_exponent, gamma = gamma) 
        environments_list.append(env)


    initial_p_z = 2 * np.ones((n_devices, ))
    while np.sum(initial_p_z) >1:
        for i in range(n_devices):
            initial_p_z[i] = random.random()

    print(initial_p_z, run_index)


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

    unique_initial_policy, unique_initial_policy_vector = find_approximated_optimal_policies(environments_list=environments_list, approximated_computation=True, p_z_list=unique_initial_p_z, n_devices=n_devices, single_core = True)
    
    unique_initial_reward = compute_total_return(environments_list, True, p_z_list=initial_p_z, n_devices = n_devices, optimal_policy_list=unique_initial_policy, optimal_policy_vector_list=unique_initial_policy_vector)
    unique_initial_reward = np.sum(unique_initial_reward)

    # we first should find a baseline value: we compute the average of 10 evaluation with random initial p_z
    _, _, reward_naive_random = naive_random_search_method(environments_list, n_devices, approximated_computation=True, single_core=True, return_list_rewards=True, fixed_number_attempts=10)
    baseline = np.mean(reward_naive_random)

    # exact calclulation
    start_time = time.time()
    optimal_p_z, optimal_total_reward, rewards_sequential_SPSA_exact = sequential_SPSA(n_devices = n_devices, k = 1, environments_list = environments_list, approximated_computation = False, return_list_rewards=True, initial_p_z = initial_p_z, initial_total_reward = unique_initial_reward, initial_policy_list=unique_initial_policy, initial_policy_vector_list=unique_initial_policy_vector, single_core=True)

    # approximate calculation
    start_time_sequential_SPSA = time.time()
    optimal_p_z, optimal_total_reward, rewards_sequential_SPSA = sequential_SPSA(n_devices = n_devices, k = 1, environments_list = environments_list, approximated_computation = True, return_list_rewards=True, initial_p_z = initial_p_z, initial_total_reward = unique_initial_reward, initial_policy_list=unique_initial_policy, initial_policy_vector_list=unique_initial_policy_vector, single_core = True)
    print(run_index, n_devices, initial_p_z, baseline, rewards_sequential_SPSA, rewards_sequential_SPSA_exact)
    return [rewards_sequential_SPSA_exact, rewards_sequential_SPSA, baseline]




def parallel_computing_handler(list_unique_initial_distributions, n_devices, n_runs):
    
    pool = mp.Pool(min(mp.cpu_count() -1, n_runs))
    list_baselines = []
    list_rewards_sequential_SPSA = []
    list_rewards_sequential_SPSA_exact = [] 

    parallel_result = pool.starmap(optimal_polling_distribution, [(list_unique_initial_distributions[run_index], n_devices, run_index) for run_index in range(n_runs)])
    
    for i in range(len(list_unique_initial_distributions)):
        attempt = parallel_result[i]
        list_rewards_sequential_SPSA_exact.append(attempt[0])
        list_rewards_sequential_SPSA.append(attempt[1]) 
        list_baselines.append(attempt[2]) 

    return list_rewards_sequential_SPSA_exact, list_rewards_sequential_SPSA, list_baselines
  
n_devices_list = [1, 3, 5, 7, 10]
n_devices_list = [3]
n_initial_distributions = 50

initial_time_all = time.time()


folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/stochastic_approximation_methods_comparison/'
if not os.path.exists(folder):
    os.makedirs(folder)



big_list_rewards_exact = []
big_list_rewards_approximated = []
big_list_baselines = []

for n_devices in n_devices_list:
    list_unique_initial_distributions = []
    while len(list_unique_initial_distributions) < n_initial_distributions:
        unique_initial_p_z = np.zeros((n_devices, ))
        for i in range(n_devices):
            unique_initial_p_z[i] = random.random()
        if np.sum(unique_initial_p_z) <= 10:
            list_unique_initial_distributions.append(unique_initial_p_z)
            

    list_rewards_sequential_SPSA_exact, list_rewards_sequential_SPSA, list_baselines = parallel_computing_handler(list_unique_initial_distributions = list_unique_initial_distributions, n_devices = n_devices, n_runs = n_initial_distributions)
    # print(list_rewards_sequential_SPSA)
    # print(list_rewards_sequential_SPSA_exact)
    big_list_rewards_exact.append(list_rewards_sequential_SPSA_exact)
    big_list_rewards_approximated.append(list_rewards_sequential_SPSA)
    big_list_baselines.append(list_baselines)

# here we need to compute the ratio between the approximate and the exact reward

# we can now save in a file the list of values (reward approximate, reward_exact, number of policy learnings and rewards' ratio)
n_devices = n_devices_list[0]
pk.dump(big_list_rewards_exact, open(folder + 'big_list_rewards_exact_' +str(n_devices) +'.dat', 'wb'))
pk.dump(big_list_rewards_approximated, open(folder + 'big_list_rewards_approximated_' +str(n_devices) +'.dat', 'wb'))
pk.dump(big_list_baselines, open(folder + 'big_list_baselines_' +str(n_devices) +'.dat', 'wb'))


final_time = time.time()


print("--- %s seconds ---" % (final_time - initial_time_all))