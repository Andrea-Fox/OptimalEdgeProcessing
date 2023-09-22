from _mdp import BatteryBuffer
from _value_iteration_helper import plot_solution, compute_optimal_solution, compute_P0, compute_P1, compute_stationary_distribution, compute_matrix_gamma
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})
import math
import os 
import sys

import multiprocessing as mp
import pickle as pk

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/average_peak_AoI/'
print(folder)

parameter_to_change = 'Harvesting_distribution'
average_peak_aoi_evolution_list = pk.load(open(folder + parameter_to_change +'/average_peak_aoi_evolution_list.dat', 'rb'))
print(average_peak_aoi_evolution_list)
colors = ['#ff595e', '#5b8e7d', '#8ac926', '#1982c4', '#6a4c93']
linestyles = ['--', '-.','-', '-', '-']

if parameter_to_change == 'p_Z':
    list_parameters_comparison = np.linspace(0, 1, 100) #[.01, .05, .1, .25, .5, .75, .999] 
elif parameter_to_change == 'Cost_distribution':
    list_parameters_comparison = [1, 2, 3, 3, 3, 3, 3] 
elif parameter_to_change == 'Harvesting_distribution':
    list_parameters_comparison = np.linspace(0, 5, 100)
elif parameter_to_change == 'Mean_symmetric_distribution':
    # these are the values of p_z that are considered 
    list_parameters_comparison = [.01, .05, .1, .25, .5, .75, .999] 
elif parameter_to_change == 'Delta':
    list_parameters_comparison = [0, 1, 2, 3, 5, 15]
elif parameter_to_change == 'Delta_evolution_p_z':
    list_parameters_comparison = np.linspace(0, 1, 100) 
    
list_delta_comparison = np.arange(1, 16) # 0, 1, 2, 3, 5, 15]
list_mean_symmetric_comparison = [1, 3, 5, 7, 8.478, 10, 15]
list_sigma_comparison = [0, 0, 0.5, 1, 3, 5, 10]                           # needed only when considering different cost distributions
list_divisor_reward_comparison = [10, 10, 10, 10, 10]          

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

if parameter_to_change == 'Cost_distribution' and len(list_sigma_comparison) == len(list_parameters_comparison):
    if 1 in list_parameters_comparison:
        plt.axhline(y = average_peak_aoi_uniform, color = 'r', linestyle = '-')
    if 2 in list_parameters_comparison:
        plt.axhline(y = average_peak_aoi_two_spikes, color = 'g', linestyle = '-')
    plt.plot(list_sigma_comparison[2:], average_peak_aoi_evolution[2:], marker = 'o', color = 'b')
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
    plt.savefig(folder + parameter_to_change + '/comparison.png')
    plt.show()
elif parameter_to_change == 'Delta_evolution_p_z':
    # for each different value of the mean we need ot print the evolution accroding to the p_Z
    for index in range(len(cost_probability_values)):
        plt.plot(list_parameters_comparison, average_peak_aoi_evolution_list[index]) #, marker = 'o')
    plt.legend(cost_probability_values)
    for index in range(len(cost_probability_values)):
        plt.scatter(list_parameters_comparison[np.argmin(average_peak_aoi_evolution_list[index])], min(average_peak_aoi_evolution_list[index]))    
    plt.title("Peak Age of Information evolution")
    plt.xlabel("p_Z")
    plt.ylabel("Peak Age of Information")
    plt.ylim([0, 20])
    # plt.savefig(folder + parameter_to_change + '/comparison.png')
    plt.show()
else:
    plt.figure(figsize=(7,6))
    for index in range(len(cost_probability_values)):
        plt.plot(list_parameters_comparison, average_peak_aoi_evolution_list[index], color = colors[index], ls = linestyles[index], linewidth=2, zorder=1) #, marker = 'o')
        # find the minimum and highlight it
     
    # plt.title("Peak Age of Information evolution")
    if parameter_to_change == 'Harvesting_distribution':
        plt.xlabel('$\lambda$')
    else:
        plt.xlabel('$p_Z$')
    plt.ylabel("Average Peak Age of Information")
    plt.ylim([0, 7])
    # plt.yticks(np.arange(0, max(max(average_peak_aoi_evolution_list[0]), max(average_peak_aoi_evolution_list[1]), max(average_peak_aoi_evolution_list[2]))+1, 1))
    if max(list_parameters_comparison) > 1:
        plt.xticks(np.arange(0, max(list_parameters_comparison)+1))
    else:
        plt.xticks(np.arange(0, 1.1, .2))
    plt.legend(["Uniform", "Binary", "Symmetric, $\mu = 1$", "Symmetric, $\mu = 4$", "Symmetric, $\mu = 8.5$"], fancybox=True, framealpha=0.5, prop={'size': 15})
    for index in range(len(cost_probability_values)):
        min_value = min(average_peak_aoi_evolution_list[index])
        min_p_z = list_parameters_comparison[np.argmin(average_peak_aoi_evolution_list[index])]
        print(min_value, min_p_z)
        plt.scatter(min_p_z, min_value, color = colors[index], zorder=2) # , marker = 'x')
    if parameter_to_change == 'p_Z':
        plt.savefig(folder + parameter_to_change + '/comparison_p_z.png')
    elif parameter_to_change == 'Harvesting_distribution':
        plt.savefig(folder + parameter_to_change + '/comparison_harvesting_rate.png')
    else:
        plt.savefig(folder + parameter_to_change + '/comparison.png')


    plt.show()

if parameter_to_change == 'Cost_distribution':
    pk.dump(list_sigma_comparison, open(folder + parameter_to_change +'/sigma_comparison.dat', 'wb'))
else:
    pk.dump(list_parameters_comparison, open(folder + parameter_to_change +'/parameters_comparison.dat', 'wb'))

