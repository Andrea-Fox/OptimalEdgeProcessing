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



parameter_to_change = 'delta'

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/optimal_polling_probability_comparison/' + parameter_to_change + '/'

plt.figure(figsize=(7, 6))
colors = ['#ff595e', '#5b8e7d', '#8ac926', '#1982c4', '#6a4c93']
linestyles = ['--', '-.','-', '-', '-']

find_all_minimums_binary = pk.load(open(folder + 'min_average_paoi_only_binary.dat', 'rb'))
find_all_minimums = pk.load(open(folder + 'min_average_paoi.dat', 'rb'))

print(find_all_minimums)
# we have to add the term 1 corresponding to delta = 0

list_min_p_z_uniform = [1]
list_min_p_z_binary = [1]
list_min_p_z_symmetric_1 = [1]
list_min_p_z_symmetric_4 = [1]
list_min_p_z_symmetric_85 = [1]
for i in range(len(find_all_minimums)):
    minimum = find_all_minimums[i][0]
    minimum_binary = find_all_minimums_binary[i][0]
    print(minimum[0], minimum[1],minimum[2],minimum[3],minimum[4])
    print('---------------------------------')
    list_min_p_z_uniform.append(minimum[0])
    list_min_p_z_binary.append(minimum_binary[0])
    list_min_p_z_symmetric_1.append(minimum[2])
    list_min_p_z_symmetric_4.append(minimum[3])
    list_min_p_z_symmetric_85.append(minimum[4])
plot_type = 'line'
if plot_type == 'scatter':
    plt.scatter(np.arange(0, 17), list_min_p_z_uniform)
    plt.scatter(np.arange(0, 17), list_min_p_z_binary)
    plt.scatter(np.arange(0, 17), list_min_p_z_symmetric_1)
    plt.scatter(np.arange(0, 17), list_min_p_z_symmetric_4)
    plt.scatter(np.arange(0, 17), list_min_p_z_symmetric_85)
elif plot_type == 'line':
    plt.plot(np.arange(0, 17), list_min_p_z_uniform, color = colors[0], ls = linestyles[0], linewidth=2, zorder=1)
    plt.plot(np.arange(0, 17), list_min_p_z_binary, color = colors[1], ls = linestyles[1], linewidth=2, zorder=1)
    plt.plot(np.arange(0, 17), list_min_p_z_symmetric_1, color = colors[2], ls = linestyles[2], linewidth=2, zorder=1)
    plt.plot(np.arange(0, 17), list_min_p_z_symmetric_4, color = colors[3], ls = linestyles[3], linewidth=2, zorder=1)
    plt.plot(np.arange(0, 17), list_min_p_z_symmetric_85, color = colors[4], ls = linestyles[4], linewidth=2, zorder=1)
plt.ylabel('Optimal value of $p_Z$')
plt.xlabel('$\delta$')
plt.legend(["Uniform", "Binary", "Symmetric, $\mu = 1$", "Symmetric, $\mu = 3$", "Symmetric, $\mu = 8.5$"], prop={'size': 15})
plt.xlim([-1, 17])
plt.ylim([-.05, 1.1])
plt.savefig(folder + 'comparison_min_p_z_delta.png')
plt.show()