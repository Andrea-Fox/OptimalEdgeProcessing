from _mdp import BatteryBuffer
from _value_iteration_helper import plot_solution, compute_optimal_solution, compute_P0, compute_P1, compute_stationary_distribution, compute_matrix_gamma
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys

import multiprocessing as mp
import pickle as pk


def plot_comparison(data, parameter_to_change, legend_list, title):
    for index in range(len(data)):
        plt.plot(data[index])
    plt.legend(legend_list,  title = parameter_to_change)
    plt.title(title)
    plt.show()

def compute_peak_AoI(parameter_to_change, M, B, h, p_z, cost_probability, reward_function, gamma, eps_0, exploration_rate, delta, mean_cost_normal_distr, index, list_parameters_comparison, cost_probability_index):
    parameter = list_parameters_comparison[index]
    disadvantage_function = (lambda e: (e<0) *(-e)**2)
    cost_probability_values = [1, 2, 3, 3, 3]
    mean_cost_probability_values = [0, 0, 1, 4, mean_cost_normal_distr]

    if parameter_to_change == "M":
        reward_function = np.arange(parameter, 0, -1)
        env = BatteryBuffer(M = parameter, B = B, h=h, p_z=p_z, cost_probability= cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_normal_distr)
    elif parameter_to_change == "B":
        env = BatteryBuffer(M = M, B = parameter, h=h, p_z=p_z, cost_probability= cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_normal_distr)
    elif parameter_to_change == "p_Z":
        env = BatteryBuffer(M = M, B = B, h=h, p_z=parameter, cost_probability= cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_probability_values[cost_probability_index])
    elif parameter_to_change == "Cost_distribution":
        if len(list_sigma_comparison) == len(list_parameters_comparison):
            # we also consider the sigma of a normal distribution
            env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = parameter, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, sigma_normal_distr=list_sigma_comparison[index], delta=delta, mean_value_normal_distr=mean_cost_normal_distr)
        else:
            env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = parameter, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_normal_distr)
    elif parameter_to_change == "gamma":
        env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = parameter, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_normal_distr)
    elif parameter_to_change == 'Harvesting_distribution':
        env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, lambda_harvesting_distribution=parameter, delta=delta, mean_value_normal_distr=mean_cost_probability_values[cost_probability_index])
    elif parameter_to_change == 'Disadvantage function':
        disadvantage_function = (lambda e: (e<0) *(-e)**parameter)
        env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_normal_distr) 
    elif parameter_to_change == 'Mean_symmetric_distribution':
        env = BatteryBuffer(M = M, B = B, h=h, p_z=parameter, cost_probability = 3, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=cost_probability) 
    elif parameter_to_change == 'Delta':
        env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=parameter, mean_value_normal_distr=mean_cost_normal_distr) 
    elif parameter_to_change == 'Delta_evolution_p_z':
        # cost probability represents the value of delta considered in the current iteration. The cost probability is always 3 (symmetric)
        env = BatteryBuffer(M = M, B = B, h=h, p_z=parameter, cost_probability = 2, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=cost_probability, mean_value_normal_distr=mean_cost_normal_distr) 
    elif parameter_to_change == 'Reward function':
        if len(list_divisor_reward_comparison) == len(list_parameters_comparison):
            reward_function = (np.arange(M, 0, -1)**parameter)/list_divisor_reward_comparison[index]
        else:
            reward_function = (np.arange(M, 0, -1)**parameter)
        env = BatteryBuffer(M = M, B = B, h=h, p_z=p_z, cost_probability = cost_probability, reward_function=reward_function, disadvantage_function = disadvantage_function, gamma = gamma, eps_0 = eps_0, exploration_rate=exploration_rate, delta=delta, mean_value_normal_distr=mean_cost_normal_distr) 
    
    # find optimal strategy
    optimal_policy_vector, optimal_policy = compute_optimal_solution(env=env, disadvantage_function=env.disadvantage_function, reward_function=env.reward_function, cost_distribution=env.p_C, harvesting_distribution = env.p_H)
    # if plot_optimal_policy:
    #     plot_solution(optimal_policy, env.M, env.B)
    if env.p_z == 1:
        average_peak_aoi = 1 + env.delta
    else:
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
        # gamma matrix
        gamma_matrix = compute_matrix_gamma(env)
        tau = np.zeros((env.B+1,))

        # steps_probability[e, k] denotes the probability of getting to state (1, e) after k (k>=1) steps after action 1,
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
        # we have the correct value of tau for each final energy level: we can finally compute the sampling rate
        normalized_stationary_probability_aoi_1 = np.zeros((env.B+1, ))
        for e in range(env.B+1):
            normalized_stationary_probability_aoi_1[e] = stationary_distribution[env.compute_index(1, e, 0)] + stationary_distribution[env.compute_index(1, e, 1)]
        normalized_stationary_probability_aoi_1 = normalized_stationary_probability_aoi_1/sum(normalized_stationary_probability_aoi_1)
        average_peak_aoi = np.dot(normalized_stationary_probability_aoi_1, tau) + 1/average_cycle_length - 1
    
    
    # print(cost_probability_index, parameter, average_peak_aoi)
    return average_peak_aoi

##################################################################################################################
pool = mp.Pool(mp.cpu_count() -2)


# Parameters to change
parameter_to_change = 'p_Z'
if parameter_to_change == 'p_Z':
    list_parameters_comparison = np.linspace(0, 1, 100) #[.01, .05, .1, .25, .5, .75, .999] 
elif parameter_to_change == 'Cost_distribution':
    list_parameters_comparison = [1, 2, 3, 3, 3, 3, 3] 
elif parameter_to_change == 'Harvesting_distribution':
    list_parameters_comparison = np.linspace(0, 5, 100) # [0, 1, 2, 3, 5]
elif parameter_to_change == 'Mean_symmetric_distribution':
    # these are the values of p_z that are considered 
    list_parameters_comparison = [.01, .05, .1, .25, .5, .75, .999] 
elif parameter_to_change == 'Delta':
    list_parameters_comparison = [0, 1, 2, 3, 5, 15]
elif parameter_to_change == 'Delta_evolution_p_z':
    list_parameters_comparison = np.linspace(0, 1, 100) #[.01, .05, .1, .25, .5, .75, .999] 
    
list_delta_comparison = np.arange(1, 16) # 0, 1, 2, 3, 5, 15]
list_mean_symmetric_comparison = [1, 3, 5, 7, 8.478, 10, 15]
list_sigma_comparison = [0, 0, 0.5, 1, 3, 5, 10]                           # needed only when considering different cost distributions
list_divisor_reward_comparison = [10, 10, 10, 10, 10]               # needed only when considering different reward functions

plot_optimal_policy = False

############################################################# 
possible_parameters_to_change = ['M', 'B', 'p_Z', 'Cost_distribution', 'gamma', 'Harvesting_distribution', 'Disadvantage function', 'Reward function', 'Mean_symmetric_distribution', 'Delta', 'Delta_evolution_p_z'] 
if parameter_to_change not in possible_parameters_to_change:
    raise ValueError('parameter_to_change value is not valid.\nThe possible options are '+ str(possible_parameters_to_change))


sampling_rate_evolution = []
legend_list = []

if parameter_to_change == "Cost distribution" and len(list_sigma_comparison) == len(list_parameters_comparison):
    for index in range(len(list_parameters_comparison)):
        if list_parameters_comparison[index] == 1:
            legend_list.append('Uniform distribution')
        elif list_parameters_comparison[index] == 2:
            legend_list.append('Two spikes distribution')
        if list_parameters_comparison[index] == 3:
            legend_list.append('$\mathcal{N}(\mu = 3, \sigma =$ ' + str(list_sigma_comparison[index]) + ')')
elif parameter_to_change == 'Cost distribution':
    for index in range(len(list_parameters_comparison)):
        if list_parameters_comparison[index] == 1:
            legend_list.append('Uniform distribution')
        elif list_parameters_comparison[index] == 2:
            legend_list.append('Two spikes distribution')
        if list_parameters_comparison[index] == 3:
            legend_list.append('$\mathcal{N}(\mu = 3, \sigma = 3 )$')
else:
    legend_list = list_parameters_comparison

# definition of the system to study (we also define the parameters that will later be overwritten by those in list_parameter_comparison)
M = 15
B = 15
gamma = 0.95
p_z = 0.05
eps0 = 0
delta = 1
h = 1
mean_cost_normal_distr = 8.478
if parameter_to_change != 'Cost_distribution' and parameter_to_change != 'Mean_symmetric_distribution' and parameter_to_change != 'Delta_evolution_p_z':
    cost_probability_values = [1, 2, 3, 3, 3]
    mean_cost_probability_values = [0, 0, 1, 4, mean_cost_normal_distr]
elif parameter_to_change == 'Mean_symmetric_distribution':
    cost_probability_values = list_mean_symmetric_comparison
elif parameter_to_change == 'Delta_evolution_p_z':
    cost_probability_values = list_delta_comparison
else:
    cost_probability_values = [1]

eps_0 = 0
exploration_rate = 0.1
lambda_harvesting_distribution = 1          # we assume that the harvesting rate always follows a poisson distribution with a variable parameter lambda (default = 1)
reward_function = np.arange(M, 0, -1)

disadvantage_function = (lambda e: (e<0) *(-e)**2)


average_peak_aoi_evolution_list = []
for cost_probability_index in range(len(cost_probability_values)):
    cost_probability = cost_probability_values[cost_probability_index]
    average_peak_aoi_evolution = np.zeros((len(list_parameters_comparison), ))

    average_peak_aoi_evolution = pool.starmap(compute_peak_AoI, [(parameter_to_change, M, B, h, p_z, cost_probability, reward_function, gamma, eps_0, exploration_rate, delta, mean_cost_normal_distr, index, list_parameters_comparison, cost_probability_index) for index in range(len(list_parameters_comparison))])
    # for index in range(len(list_parameters_comparison)):
    #     # generate the environment according to the parameters that change
    #     parameter = list_parameters_comparison[index]
    #     print(parameter, cost_probability_index)   
    
    average_peak_aoi_evolution_list.append(average_peak_aoi_evolution)


if parameter_to_change != 'Cost_distribution' and parameter_to_change != 'Mean_symmetric_distribution':
    a =0



folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/average_peak_AoI/'
print(folder)

save_fig = False

if parameter_to_change == 'Cost_distribution' and len(list_sigma_comparison) == len(list_parameters_comparison):
    if 1 in list_parameters_comparison:
        plt.axhline(y = average_peak_aoi_uniform, color = 'r', linestyle = '-')
    if 2 in list_parameters_comparison:
        plt.axhline(y = average_peak_aoi_two_spikes, color = 'g', linestyle = '-')
    plt.plot(list_sigma_comparison[2:], average_peak_aoi_evolution[2:]) # , marker = 'o', color = 'b')
    if 1 in list_parameters_comparison and 2 in list_parameters_comparison:
        plt.legend(['Uniform distribution', 'Two spikes distribution', 'Normal distribution'])
    plt.title("Peak Age of Information evolution")
    plt.xlabel("$\sigma$")
    plt.ylabel("Peak Age of Information")
    plt.yticks(np.arange(0, max(average_peak_aoi_two_spikes, average_peak_aoi_two_spikes)+1, 1))
    if max(list_parameters_comparison) > 1:
        plt.xticks(np.arange(0, max(list_sigma_comparison)+1))
    else:
        plt.xticks(np.arange(0, 1.1, .1))
    if save_fig:
        plt.savefig(folder + parameter_to_change + '/comparison.png')
    plt.show()
elif parameter_to_change == 'Mean_symmetric_distribution':
    # for each different value of the mean we need ot print the evolution accroding to the p_Z
    for index in range(len(list_mean_symmetric_comparison)):
        plt.plot(list_parameters_comparison, average_peak_aoi_evolution_list[index], marker = 'o')
    plt.legend(list_mean_symmetric_comparison)
    plt.title("Peak Age of Information evolution")
    plt.xlabel("p_Z")
    plt.ylabel("Peak Age of Information")
    plt.ylim([0, 10])
    if save_fig:
        plt.savefig(folder + parameter_to_change + '/comparison.png')
    plt.show()
elif parameter_to_change == 'Delta_evolution_p_z':
    # for each different value of the mean we need ot print the evolution accroding to the p_Z
    for index in range(len(cost_probability_values)):
        plt.plot(list_parameters_comparison, average_peak_aoi_evolution_list[index]) # , marker = 'o')
    plt.legend(cost_probability_values)
    plt.title("Peak Age of Information evolution")
    plt.xlabel("p_Z")
    plt.ylabel("Peak Age of Information")
    plt.ylim([0, 20])
    if save_fig:
        plt.savefig(folder + parameter_to_change + '/comparison.png')
    plt.show()
else:
    for index in range(len(cost_probability_values)):
        plt.plot(list_parameters_comparison, average_peak_aoi_evolution_list[index], marker = 'o')
    plt.title("Peak Age of Information evolution")
    if parameter_to_change == 'Harvesting_distribution':
        plt.xlabel('$\lambda$')
    else:
        plt.xlabel(parameter_to_change)
    plt.ylabel("Peak Age of Information")
    plt.yticks(np.arange(0, max(max(average_peak_aoi_evolution_list[0]), max(average_peak_aoi_evolution_list[1]), max(average_peak_aoi_evolution_list[2]))+1, 1))
    if max(list_parameters_comparison) > 1:
        plt.xticks(np.arange(0, max(list_parameters_comparison)+1))
    else:
        plt.xticks(np.arange(0, 1.1, .1))
    plt.legend(["Uniform distribution", "Two spikes distribution", "Symmetric distribution, $\mu = 1$", "Symmetric distribution, $\mu = 4$", "Symmetric distribution, $\mu = 8.5$"])
    if save_fig:
        plt.savefig(folder + parameter_to_change + '/comparison.png')
    plt.show()

if parameter_to_change == 'Cost_distribution':
    pk.dump(list_sigma_comparison, open(folder + parameter_to_change +'/sigma_comparison.dat', 'wb'))
else:
    pk.dump(list_parameters_comparison, open(folder + parameter_to_change +'/parameters_comparison.dat', 'wb'))

if parameter_to_change == 'Cost_distribution' and 1 in list_parameters_comparison:
    pk.dump(average_peak_aoi_uniform, open(folder + parameter_to_change +'/average_peak_aoi_uniform.dat', 'wb'))
if parameter_to_change == 'Cost_distribution' and 2 in list_parameters_comparison:
    pk.dump(average_peak_aoi_two_spikes, open(folder + parameter_to_change +'/average_peak_aoi_two_spikes.dat', 'wb'))

pk.dump(average_peak_aoi_evolution_list, open(folder + parameter_to_change +'/average_peak_aoi_evolution_list.dat', 'wb'))

