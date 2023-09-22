
import pickle as pk
import matplotlib.pyplot as plt 
import numpy as np
import math
import os
import sys


folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/stochastic_approximation_methods_comparison/'
if not os.path.exists(folder):
    os.makedirs(folder)


n_devices_list = [1, 3, 5, 7, 10]

list_rewards_ratios = []
list_n_policy_learning = []

# n_devices_list = [3] # , 3, 5, 7, 10]

for n_index in range(len(n_devices_list)):
    n_devices = n_devices_list[n_index]
    # print(list_rewards_sequential_SPSA_exact)
    if n_devices == 3:
        list_rewards_sequential_SPSA = pk.load(open(folder + 'big_list_rewards_approximated_' +str(n_devices) +'.dat', 'rb'))
        list_rewards_sequential_SPSA_exact = pk.load(open(folder + 'big_list_rewards_exact_' +str(n_devices) +'.dat', 'rb'))
        list_len = pk.load(open(folder + 'list_len_' +str(n_devices) +'_devices.dat', 'rb'))[0]
        list_baseline = pk.load(open(folder + 'big_list_baselines_' +str(n_devices) +'.dat', 'rb'))
        print(list_rewards_sequential_SPSA)
    else:
        list_rewards_sequential_SPSA = pk.load(open(folder + 'big_list_rewards_approximated_' +str(n_devices) +'.dat', 'rb'))[0]
        list_rewards_sequential_SPSA_exact = pk.load(open(folder + 'big_list_rewards_exact_' +str(n_devices) +'.dat', 'rb'))[0]
        list_baseline = pk.load(open(folder + 'big_list_baselines_' +str(n_devices) +'.dat', 'rb'))[0]

    
    rewards_ratios = []
    n_policy_learning = []

    # for each attempt we consider the highest value for both the exact and the approximate quantity and then compute the ratio (considering also the baseline)
    for attempt in range(50):
        if n_devices == 3:
            approximated_return = list_rewards_sequential_SPSA[attempt]
            n_policy_learning.append(len(list_len[attempt]))
            exact_return = list_rewards_sequential_SPSA_exact[attempt]
            baseline_value = list_baseline[attempt]
        else:
            approximated_return = list_rewards_sequential_SPSA[attempt][-1]
            n_policy_learning.append(len(list_rewards_sequential_SPSA[attempt]))
            exact_return = list_rewards_sequential_SPSA_exact[attempt][-1]
            baseline_value = list_baseline[attempt]
        

        rewards_ratios.append(max(0, min(1, float((approximated_return - baseline_value)/(exact_return - baseline_value)))))
    
    list_rewards_ratios.append(rewards_ratios)
    list_n_policy_learning.append(n_policy_learning)

    # print(rewards_ratios)
    # print(n_policy_learning)


for i in range(len(list_rewards_ratios)):
    # print(list_rewards_ratios[i])
    print(str(np.mean(list_rewards_ratios[i])) + ' \\pm ' + str(np.std(list_rewards_ratios[i])) )
    print(str(np.mean(list_n_policy_learning[i])) + ' \\pm ' + str(np.std(list_n_policy_learning[i])) )
    print('-----------------------------------------------------------------')



legend = []
for n_device in n_devices_list:
    legend.append('n = ' + str(n_device))


plt.boxplot(list_rewards_ratios)
# plt.ylim([0, 1])
plt.title('Ratio rewards')
plt.xticks(ticks=np.arange(1, len(n_devices_list)+1), labels=legend)
plt.savefig(folder + 'comparison_ratio_boxplot.png')
plt.show()

plt.boxplot(list_n_policy_learning)
plt.xticks(ticks=np.arange(1, len(n_devices_list)+1), labels=legend)
plt.title('Policy learning operations')
plt.ylim([0, 10])
plt.savefig(folder + 'comparison_n_policy_learning_boxplot.png')
plt.show()
