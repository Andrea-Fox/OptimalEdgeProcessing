# we need to find the minimum in the peak age of information
# we know that the curve is unimodal, therefore finding the minimum requires some numerocal analysis algorithm

# for each value of mean_cost_symmetric_distribution, we value of p_Z corresponding to the minimum peak age of information

# we plot: a curve with the evolution of min p_Z for each value of delta. We have a line for the other two cost distributions
# y axis: min p_Z, x_axis value of mean_cost_symmetric_distribution

# to find the minimum we can recycle SPSA algorithm

import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import random
import numpy as np
import multiprocessing as mp

from _mdp import BatteryBuffer
from _value_iteration_helper import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

import pickle as pk


def compute_mean_value_cost_distr(env):
    mean_value = 0 
    for i in range(0, env.B+3):
        mean_value += env.p_C[i]*i
    return mean_value

def compute_average_peak_aoi(env, fixed_p_z):
    # find optimal strategy
    if fixed_p_z<1:
        plot_optimal_policy = False
        env.p_z = fixed_p_z
        optimal_policy_vector, optimal_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H)
        if plot_optimal_policy:
            plot_solution(optimal_policy, env.M, env.B)
        # define markov chain given by optimal policy
        transition_matrix_markov_chain = np.zeros((env.num_states, env.num_states))
        # transition matrix of action a=0
        P0 = compute_P0(env, env.p_H)
        # transition matrix of action a=1
        P1 = compute_P1(env, env.p_C, env.p_H)
        for state in range(env.num_states):
            action = optimal_policy_vector[state]
            if action == 0:
                for final_state in range(env.num_states):     
                    transition_matrix_markov_chain[state, final_state] = P0[state, final_state]
            elif action == 1:
                for final_state in range(env.num_states):     
                    transition_matrix_markov_chain[state, final_state] = P1[state, final_state]
        # find stationary distribution of the Markov chain
        stationary_distribution = compute_stationary_distribution(transition_matrix_markov_chain)

        # computation of the average cycle length
        average_cycle_length = 0
        for index_cycle_length in range(env.num_states):
            if optimal_policy_vector[index_cycle_length]:
                average_cycle_length += stationary_distribution[index_cycle_length]
        # definition of p_E as the probability of doing action 1 at energy level e
        p_E = np.zeros(env.B+1)
        for e in range(env.B+1):
            # compute the leftmost state at energy e that does action 1
            minimum_x = env.M
            for x in range(env.M):
                if optimal_policy[x, e] == 1:
                    minimum_x = min(x, minimum_x)
            # print(minimum_x, e)
            p_E[e] = stationary_distribution[env.compute_index(minimum_x+1, e, 0)]
        p_E = p_E/sum(p_E)
        # print(p_E)
        # print(sum(p_E))
        # gamma matrix
        gamma_matrix = compute_matrix_gamma(env)
        # print(gamma_matrix)
        # for k in range(env.B+1):
        #     print(sum(gamma_matrix[k, :]))

        tau = np.zeros((env.B+1,))
        # steps_probability[e, k] indicates the probability of getting to state (1, e) after k (k>=1) steps after action 1,
        # with k=1 representing that the action is completed without additional harvesting. We should have that steps_probability[e, :] is 
        # a probability distribution
        steps_probability = np.zeros((env.B+1, max(env.B+1, env.delta*(env.B+1))))

        # if we set env.p_z == 1 from the beginning, problems related to the transition probabilities arise. In particular, the transition 
        # probabilities starting from states with for z=0 make no sense. With these lines we allow the computation of the transition 
        # probabilities with a very high value of env.p_z, but consider env.p_z = 1 when computing the time spent to complete a transition.
        # By doing so, we obtain exactly that when env.p_z = 1, the sojourn time is always equal to 1+env.delta 
        # if env.p_z >= 0.99:
        #     env.p_z = 1
        for final_energy in range(env.B+1):
            # final_energy indicates the arrival energy
            # first we compute the probability of arriving in state (1, e, z) after the server completed its action. 
            # Note how we need to take into account the time it takes to complete the offloading
            for e in range(env.B+1):
                steps_probability[final_energy, 1+env.delta] += env.p_z * p_E[e] * env.p_H[final_energy - e]
            # then we compute the term corresponding to the completion of the action without needing an additional harvesting
            for e in range(env.B+1):
                for r in range(1, env.B+1):
                    try:
                        # it is possible that p_C is not defined for certain values of [e + r - final_energy]
                        # tau[final_energy] += p_E[e] * gamma_matrix[1, r] * env.p_C[e+r - final_energy]
                        steps_probability[final_energy, 1] += (1-env.p_z) * p_E[e] * gamma_matrix[1, r] * env.p_C[e + r - final_energy]
                    except:
                        a = 0
            for k in range(2, env.B):
                for e in range(env.B+1):
                    for r in range(k-1, env.B * (k-1)):
                        for c in range(e+r+1, 20):
                            try:
                                # tau[final_energy] += k * p_E[e] * gamma_matrix[k-1, r] * env.p_C[c] * env.p_H[final_energy + c - r - e]
                                steps_probability[final_energy, k] += (1-env.p_z) * p_E[e] * gamma_matrix[k-1, r] * env.p_C[c] * env.p_H[final_energy + c - r - e]
                            except:
                                a = 0
        for final_energy in range(env.B+1):
            normalized_probabilities = steps_probability[final_energy, :]/ sum(steps_probability[final_energy, :])
            # print(normalized_probabilities)
            tau[final_energy] = np.dot(np.arange(steps_probability.shape[1]), normalized_probabilities)
        # print(tau)
        # we have the correct value of tau for each final energy level: we can finally compute the sampling rate
        normalized_stationary_probability_aoi_1 = np.zeros((env.B+1, ))
        for e in range(env.B+1):
            normalized_stationary_probability_aoi_1[e] = stationary_distribution[env.compute_index(1, e, 0)] + stationary_distribution[env.compute_index(1, e, 1)]
        normalized_stationary_probability_aoi_1 = normalized_stationary_probability_aoi_1/sum(normalized_stationary_probability_aoi_1)
        average_peak_aoi = np.dot(normalized_stationary_probability_aoi_1, tau) + 1/average_cycle_length - 1
    else:
        average_peak_aoi = 1 + env.delta
    return average_peak_aoi


def projection_p_z(p_z):
    return min(max(0, p_z), 1)


def find_min_p_Z(a, mean_cost_symmetric_distribution):
    M = 15
    B = 15
    gamma = 0.95
    p_z = 0.05
    eps0 = 0
    delta = 1
    h = 1
    lambda_harvesting_distribution = 1          # we assume that the harvesting rate always follows a poisson distribution with a variable parameter lambda (default = 1)
    reward_function = np.arange(M, 0, -1)
    cost_probability_list = 3 # [1, 2, 3]
    disadvantage_function_exponent = 4

    env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability= cost_probability_list, reward_function=reward_function, disadvantage_function_exponent= disadvantage_function_exponent, gamma = gamma, delta=delta, mean_value_normal_distr=mean_cost_symmetric_distribution)
    env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
    
    p_z = random.random() # 0.75 + 0.02 * mean_cost_symmetric_distribution 
    p_z = projection_p_z(p_z)
    old_p_z = float(p_z)
    optimal_average_paoi = compute_average_peak_aoi(env, p_z)

    return 0







def find_min_p_Z_SPSA(parameter_to_change, parameter, only_binary):

    perturbation_variance = 0.05

    random.seed(int(parameter))

    parameter = int(parameter)
    print(parameter)

    M = 15
    B = 15
    gamma = 0.95
    p_z = 0.05
    eps0 = 0
    
    h = 1
    lambda_harvesting_distribution = 1          # we assume that the harvesting rate always follows a poisson distribution with a variable parameter lambda (default = 1)
    reward_function = np.arange(M, 0, -1)
    if only_binary:
        cost_probability_list = [2]
    else:
        cost_probability_list = [1, 2, 3, 3, 3]

    delta = 1
    mean_cost_symmetric_distribution_list = [-1, -1, 1, 4, 8.458]
    disadvantage_function_exponent = 4

    optimal_min_p_z_list = []
    optimal_min_PAoI_list = []

    for index in range(len(cost_probability_list)):
        if parameter_to_change == 'mean_cost':
            env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability= cost_probability_list[index], reward_function=reward_function, disadvantage_function_exponent= disadvantage_function_exponent, gamma = gamma, delta=delta, mean_value_normal_distr=parameter)
        elif parameter_to_change == 'delta':
            env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability= cost_probability_list[index], reward_function=reward_function, disadvantage_function_exponent= disadvantage_function_exponent, gamma = gamma, delta=parameter, mean_value_normal_distr=mean_cost_symmetric_distribution_list[index])

        env.disadvantage_function = (lambda e: (e<0) *(-e)**env.disadvantage_function_exponent)
        # print(env.p_C)

        if not only_binary:
            weight_function_ascent = (lambda x: 1/((((x // 3) +3))**2))
            weight_function_perturbation = (lambda x: 1/(((x // 3) +3))**2)
        else:
            weight_function_ascent = (lambda x: 1/((( x+4))**2))
            weight_function_perturbation = (lambda x: 1/(( x + 5))**2)


        if parameter == 12:
            max_steps_from_last_update = 5
        else:
            max_steps_from_last_update = 5


        verbose = False
        eps = 0.001
        done = False
        SPSA_steps = 1
        perturbation_variance = 0.05
        steps_from_last_update = 0

        # we repeat the same process several times and for each attempt (i.e. parameter) we choose only the value of p_z corresponding to the lowest value of PAoI
        if parameter == 12:
            n_different_attempts = 5
        else:
            n_different_attempts = 5

        optimal_min_p_z = 0
        optimal_min_PAoI = math.inf
        previous_gradient = 0

        for attempt_index in range(n_different_attempts):
            # print(p_z)
            p_z = (0.25 - random.random()/20) * attempt_index
            p_z = projection_p_z(p_z)
            old_p_z = float(p_z)
            optimal_average_paoi = compute_average_peak_aoi(env, p_z)

            if verbose:
                print('Initial p_Z = ', p_z)
                print('Initial average PAoI =', optimal_average_paoi)    



            while not done:
                if verbose:
                    print('Value of p_Z = ', p_z)
                perturbation = math.fabs(random.gauss(0, perturbation_variance))
                p_z_positive_perturbation = p_z + weight_function_perturbation(SPSA_steps) * perturbation
                p_z_negative_perturbation = p_z - weight_function_perturbation(SPSA_steps) * perturbation
                p_z_positive_perturbation = projection_p_z(p_z_positive_perturbation)
                p_z_negative_perturbation = projection_p_z(p_z_negative_perturbation)

                # computation of g: first we need to compute the value of the average reward for the new perturbated vectors
                positive_perturbation_average_paoi = compute_average_peak_aoi(env, p_z_positive_perturbation)
                negative_perturbation_average_paoi = compute_average_peak_aoi(env, p_z_negative_perturbation)


                if verbose:
                    print('perturbation = ' + str(weight_function_perturbation(SPSA_steps) * perturbation))
                    print('positive perturbation projection = ' + str(p_z_positive_perturbation) + '\t' + str(positive_perturbation_average_paoi))
                    print('negative perturbation projection = ' + str(p_z_negative_perturbation) + '\t' + str(negative_perturbation_average_paoi))


                gradient_estimate = (positive_perturbation_average_paoi - negative_perturbation_average_paoi)/(2 * weight_function_perturbation(SPSA_steps)*perturbation)
                weighted_gradient_estimate = weight_function_ascent(SPSA_steps) * gradient_estimate
                if verbose:
                    print('gradient estimate = ' + str(gradient_estimate))
                    print('weighted gradient estimate = ' + str(weighted_gradient_estimate))

                # we modify p_Z in order to move towards the minimum value (opposite to the gradient estimate)
                p_z = projection_p_z(old_p_z - weighted_gradient_estimate) 

                new_average_paoi = compute_average_peak_aoi(env, p_z)

                if only_binary:
                    # we increment SPSA_steps if we have overcome the min_point
                    if previous_gradient * weighted_gradient_estimate < 0: # optimal_average_paoi:
                        SPSA_steps += 1
                    else:
                        old_p_z = float(p_z)
                        optimal_average_paoi = new_average_paoi
                        steps_from_last_update = 0
                    if math.fabs(weighted_gradient_estimate) < eps:
                        done = True 
                    previous_gradient = float(weighted_gradient_estimate)
                else:
                    if new_average_paoi < optimal_average_paoi:
                        # we have improved and need to update the optimal p_Z
                        SPSA_steps += 1
                        if verbose:
                            print(SPSA_steps, p_z, new_average_paoi, ' Updated values')
                        # if the update is small, end the process
                        if math.fabs(weighted_gradient_estimate) < eps:
                            done = True 
                        old_p_z = float(p_z)
                        optimal_average_paoi = new_average_paoi
                        steps_from_last_update = 0
                    else:
                        if verbose:
                            print(SPSA_steps, p_z, new_average_paoi)
                        p_z = old_p_z

                steps_from_last_update += 1
                if steps_from_last_update >= max_steps_from_last_update:
                    SPSA_steps += 1

                # if gradient_estimate > 0:
                #     SPSA_steps += 1

                if SPSA_steps >= 5:
                    done = True
                if verbose:
                    print('------------------------------------------')
            mean_value_cost_distr = compute_mean_value_cost_distr(env)
            # old_p_z, optimal_average_paoi
            if optimal_average_paoi < optimal_min_PAoI:
                optimal_min_p_z = old_p_z
                optimal_min_PAoI = optimal_average_paoi
        print(SPSA_steps, optimal_min_p_z, optimal_min_PAoI, parameter, cost_probability_list[index], mean_cost_symmetric_distribution_list[index])
        optimal_min_p_z_list.append(optimal_min_p_z)
        optimal_min_PAoI_list.append(optimal_min_PAoI)
    
    return optimal_min_p_z_list, optimal_min_PAoI_list


pool = mp.Pool(mp.cpu_count() -3)


# list_mean_symmetric_comparison = np.empty(15)
# list_mean_symmetric_comparison.fill(3)

save_data = True
plot = True
only_binary = True


parameter_to_change = 'delta'

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/optimal_polling_probability_comparison/' + parameter_to_change + '/'
if not os.path.exists(folder):
    os.makedirs(folder)


if parameter_to_change == 'mean_cost':    
    list_parameter_to_change = np.arange(1, 15)
elif parameter_to_change == 'delta':
    list_parameter_to_change = np.arange(1, 17)


print(list_parameter_to_change)
print(list_parameter_to_change)
find_all_minimums = pool.starmap(find_min_p_Z_SPSA, [(parameter_to_change, parameter, only_binary) for parameter in list_parameter_to_change ])

print(find_all_minimums)
if save_data:
    if only_binary:
        pk.dump(find_all_minimums, open(folder + 'min_average_paoi_only_binary.dat', 'wb'))
    else:
        pk.dump(find_all_minimums, open(folder + 'min_average_paoi.dat', 'wb'))


