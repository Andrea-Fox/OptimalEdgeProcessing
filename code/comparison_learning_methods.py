from _mdp import BatteryBuffer
from _learning_methods import *
from _value_iteration_helper import *

import random

import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import math
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

import pickle as pk

def compute_ratio(learning_value, baseline_value, optimal_policy_value):
    return min(1, max(0, (learning_value-baseline_value)/(optimal_policy_value-baseline_value)))

def compute_mean_ratio_learning_method(learning_results, baseline_results, optimal_policy_results):
    n_runs = len(learning_results)
    # check if the lengths are all the same
    if len(learning_results) != len(baseline_results) or len(learning_results)!= len(optimal_policy_results) or len(baseline_results) != len(optimal_policy_results):
        ValueError('Different vectors lengths')
    
    n_evaluations_per_run = len(learning_results[0])
    ratio_table = np.zeros((n_runs, n_evaluations_per_run)) 

    for i in range(n_runs):
        for j in range(n_evaluations_per_run):
            ratio_table[i, j] = compute_ratio(learning_results[i][j], baseline_results[i], optimal_policy_results[i])

    mean_ratio = np.zeros((n_evaluations_per_run, ))
    for j in range(1, n_evaluations_per_run):
        mean_ratio[j] = np.mean(ratio_table[:, j])    

    sd_ratio = np.zeros((n_evaluations_per_run, ))
    for j in range(1, n_evaluations_per_run):
        sd_ratio[j] = math.sqrt( np.mean( (ratio_table[:, j] - mean_ratio[j])**2  ) )

    return mean_ratio, sd_ratio

def parallel_comparison_learning_methods(index, num_episodes_learning, optimized_hyperparameters, random_environments, n_step_q_learning_flag = True, threshold_q_learning_flag = True, stairway_q_learning_flag = True, reinforce_flag = True, plot_solution = False):

    # n_step_q_learning_flag = False
    # threshold_q_learning_flag = False
    # stairway_q_learning_flag = False


    # first we need to define whether the parameter are random or not
    verbose = False
    if random_environments:
        random.seed(index)
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
        M = 15
        B = 15
        num_states = M*(B+1)*2
        gamma = 0.95
        p_z = 0.05
        eps0 = 0
        delta = 1
        disadvantage_function = (lambda e: (e<0) *(-e)**2)
        reward_function = np.arange(M, 0, -1)**1
        h = 1
        cost_probability = 3
        mean_cost = 4
        eps_0 = 0
        exploration_rate = 0.1
        env = BatteryBuffer(M = M, B = B, h=1, p_z=p_z, cost_probability= cost_probability, mean_value_normal_distr= mean_cost, reward_function = reward_function, disadvantage_function_exponent = disadvantage_function_exponent, gamma = gamma) 


    # baseline
    if random_environments or index == 0:
        baseline_policy_vector = np.zeros((env.num_states, ))
        for state_index in range(env.compute_index(env.M, env.B, 0), env.num_states):
            baseline_policy_vector[state_index] = 1
    
        baseline_policy = np.zeros((env.M, env.B + 1))
        baseline_policy[env.M-1, env.B] = 1
        baseline_discounted_reward = exact_discounted_reward_function(env = env, optimal_policy_vector=baseline_policy_vector, optimal_policy=baseline_policy)

    else:
        baseline_discounted_reward = None
    if verbose:
        print(baseline_discounted_reward)

    # compute the optimal policy and discounted reward. If random_parameter == False, then we compute it only the first time
    if random_environments or index == 0:
        optimal_policy_vector, optimal_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H)
        if plot_solution:
            plot_solution(optimal_policy, env.M, env.B)
        discounted_reward_optimal_policy = exact_discounted_reward_function(env, optimal_policy_vector, optimal_policy)
    else:
        discounted_reward_optimal_policy = None
    if verbose:
        print(discounted_reward_optimal_policy)

    n_attempts_non_optimized_parameters_q_learning = 5
    learning_rate_non_optimized_hyperparameters_q_learning = [.05, .5, .5, .25, .25]
    beta_non_optimized_hyperparameters_q_learning = [.75, .51, .75, .9, .51]
    n_non_optimized_hyperparameters_q_learning = [1, 3, 1, 1, 1]


    n_attempts_non_optimized_parameters_reinforce = 11
    policy_parameter_non_optimized_reinforce = [1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 10]
    learning_rate_constant_non_optimized_reinforce = [.01, .1, .05, .01, .1, .05, .01, .1, .05, .01, .01]
    learning_rate_exponent_non_optimized_reinforce = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    # learning methods 
    if n_step_q_learning_flag:
        if optimized_hyperparameters:
            learning_rate = .05
            beta = .75
            n = 1
            _, _, discounted_rewards_n_step_q_learning = n_step_q_learning_comparison(env = env, num_episodes = num_episodes_learning, learning_rate= learning_rate, discount_factor = 0.9, n = n, beta = beta)
        else:
            discounted_rewards_list = []
            best_rewards_list = []
            for attempt_index in range(n_attempts_non_optimized_parameters_q_learning):
                _, best_reward, discounted_reward = n_step_q_learning_comparison(env = env, num_episodes = num_episodes_learning, learning_rate= learning_rate_non_optimized_hyperparameters_q_learning[attempt_index], n = n_non_optimized_hyperparameters_q_learning[attempt_index], beta= beta_non_optimized_hyperparameters_q_learning[attempt_index])
                best_rewards_list.append(best_reward)
                discounted_rewards_list.append(discounted_reward)
            discounted_rewards_n_step_q_learning = discounted_rewards_list[np.argmax(best_rewards_list)]
    else:
        discounted_rewards_n_step_q_learning = None
    if verbose:
        print(discounted_rewards_n_step_q_learning)

    if threshold_q_learning_flag:
        if optimized_hyperparameters:
            learning_rate = .5
            beta = .51
            n = 3
            _, _, discounted_rewards_threshold_q_learning = threshold_q_learning(env = env, num_episodes = num_episodes_learning, learning_rate= learning_rate, discount_factor = 0.9, n = n, beta = beta)
        else:
            discounted_rewards_list = []  
            best_rewards_list = []
            for attempt_index in range(n_attempts_non_optimized_parameters_q_learning):
                _, best_reward, discounted_reward = threshold_q_learning(env = env, num_episodes = num_episodes_learning, learning_rate= learning_rate_non_optimized_hyperparameters_q_learning[attempt_index], n = n_non_optimized_hyperparameters_q_learning[attempt_index], beta= beta_non_optimized_hyperparameters_q_learning[attempt_index])
                best_rewards_list.append(best_reward)
                discounted_rewards_list.append(discounted_reward)
            discounted_rewards_threshold_q_learning = discounted_rewards_list[np.argmax(best_rewards_list)]
    else:
        discounted_rewards_threshold_q_learning = None
    if verbose:
        print(discounted_rewards_threshold_q_learning)

    if stairway_q_learning_flag:
        if optimized_hyperparameters:
            learning_rate = .5
            beta = .75
            n = 1
            _, _, discounted_rewards_stairway_q_learning = stairway_q_learning(env = env, num_episodes = num_episodes_learning, learning_rate=learning_rate, beta = beta, n = n)
        else:
            discounted_rewards_list = []  
            best_rewards_list = []
            for attempt_index in range(n_attempts_non_optimized_parameters_q_learning):
                _, best_reward, discounted_reward = stairway_q_learning(env = env, num_episodes = num_episodes_learning, learning_rate= learning_rate_non_optimized_hyperparameters_q_learning[attempt_index], n = n_non_optimized_hyperparameters_q_learning[attempt_index], beta= beta_non_optimized_hyperparameters_q_learning[attempt_index])
                best_rewards_list.append(best_reward)
                discounted_rewards_list.append(discounted_reward)
            discounted_rewards_stairway_q_learning = discounted_rewards_list[np.argmax(best_rewards_list)]
    else:
        discounted_rewards_stairway_q_learning = None    
    if verbose:
        print(discounted_rewards_stairway_q_learning)

    if reinforce_flag:
        if optimized_hyperparameters:
            policy_parameter = 5
            learning_rate_constant = .01
            learning_rate_exponent = 2
            _, discounted_rewards_reinforce = reinforce(env = env, policy_parameter = policy_parameter, learning_rate_constant = learning_rate_constant, learning_rate_exponent = learning_rate_exponent, n_episodes = num_episodes_learning)
        else:
            discounted_rewards_list = []  
            best_rewards_list = []
            for attempt_index in range(n_attempts_non_optimized_parameters_reinforce):
                best_reward, discounted_reward = reinforce(env = env, policy_parameter = policy_parameter_non_optimized_reinforce[attempt_index], learning_rate_constant = learning_rate_constant_non_optimized_reinforce[attempt_index], learning_rate_exponent = learning_rate_exponent_non_optimized_reinforce[attempt_index], n_episodes = num_episodes_learning)
                best_rewards_list.append(best_reward)
                discounted_rewards_list.append(discounted_reward)
            discounted_rewards_reinforce = discounted_rewards_list[np.argmax(best_rewards_list)]
    else:
        discounted_rewards_reinforce = None    
    if verbose:
        print(discounted_rewards_reinforce)

    print('Completed ' +str(index))     
    return discounted_rewards_n_step_q_learning, discounted_rewards_threshold_q_learning, discounted_rewards_stairway_q_learning, discounted_rewards_reinforce, baseline_discounted_reward, discounted_reward_optimal_policy  


pool = mp.Pool(mp.cpu_count() -3)
n_attempts = 20
save_data = True
plot_results = True


# definition of the parameters for computing the rewards
optimized_hyperparameters = False
random_environments = True
num_episodes = 3001

# computation of all the rewards
results_list = pool.starmap(parallel_comparison_learning_methods, [(index, num_episodes, optimized_hyperparameters, random_environments) for index in range(n_attempts)])


# formatting of the data to compute the ratios
discounted_rewards_n_step_q_learning = []
discounted_rewards_threshold_q_learning = []
discounted_rewards_stairway_q_learning = []
discounted_rewards_reinforce = []
baseline_values = []
discounted_reward_optimal_policy = []

for result in results_list:
    discounted_rewards_n_step_q_learning.append(result[0])
    discounted_rewards_threshold_q_learning.append(result[1])
    discounted_rewards_stairway_q_learning.append(result[2])
    discounted_rewards_reinforce.append(result[3])
    if isinstance(result[4], type(None)):
        baseline_values.append(baseline_values[0])
    else:
        baseline_values.append(result[4])
    if isinstance(result[5], type(None)):
        discounted_reward_optimal_policy.append(discounted_reward_optimal_policy[0])
    else:
        discounted_reward_optimal_policy.append(result[5])


    
# computation of the ratio vectors
ratio_n_step_q_learning_mean, ratio_n_step_q_learning_sd = compute_mean_ratio_learning_method(discounted_rewards_n_step_q_learning, baseline_values, discounted_reward_optimal_policy)
ratio_threshold_q_learning_mean, ratio_threshold_q_learning_sd = compute_mean_ratio_learning_method(discounted_rewards_threshold_q_learning, baseline_values, discounted_reward_optimal_policy)
ratio_stairway_q_learning_mean, ratio_stairway_q_learning_sd = compute_mean_ratio_learning_method(discounted_rewards_stairway_q_learning, baseline_values, discounted_reward_optimal_policy)
ratio_reinforce_mean, ratio_reinforce_sd = compute_mean_ratio_learning_method(discounted_rewards_reinforce, baseline_values, discounted_reward_optimal_policy)

# save the data

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/learning_methods_comparison/' + parameter_to_change + '/'
if not os.path.exists(folder):
    os.makedirs(folder)

if save_data:
    if optimized_hyperparameters:
        folder += 'optimized_hyperparameters/'
    else:
        folder += 'non_optimized_hyperparameters/'
    pk.dump(discounted_rewards_n_step_q_learning, open(folder + 'discounted_rewards_n_step_q_learning.dat', 'wb'))
    pk.dump(discounted_rewards_threshold_q_learning, open(folder + 'discounted_rewards_threshold_q_learning.dat', 'wb'))
    pk.dump(discounted_rewards_stairway_q_learning, open(folder + 'discounted_rewards_stairway_q_learning.dat', 'wb'))
    pk.dump(discounted_rewards_reinforce, open(folder + 'discounted_rewards_reinforce.dat', 'wb'))
    pk.dump(baseline_values, open(folder + 'baseline_values.dat', 'wb'))
    pk.dump(discounted_reward_optimal_policy, open(folder + 'discounted_reward_optimal_policy.dat', 'wb'))

    pk.dump(ratio_n_step_q_learning_mean, open(folder + 'ratio_n_step_q_learning_mean.dat', 'wb'))
    pk.dump(ratio_threshold_q_learning_mean, open(folder + 'ratio_threshold_q_learning_mean.dat', 'wb'))
    pk.dump(ratio_stairway_q_learning_mean, open(folder + 'ratio_stairway_q_learning_mean.dat', 'wb'))
    pk.dump(ratio_reinforce_mean, open(folder + 'ratio_reinforce_mean.dat', 'wb'))


# plot 
if plot_results:
    plt.plot(ratio_n_step_q_learning_mean)
    plt.plot(ratio_threshold_q_learning_mean)
    plt.plot(ratio_stairway_q_learning_mean)
    plt.plot(ratio_reinforce_mean)
    plt.show()

