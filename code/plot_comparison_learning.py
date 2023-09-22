import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys

import pickle as pk

def compute_ratio(learning_value, baseline_value, optimal_policy_value):
    return min(1, max(0, (learning_value-baseline_value)/(optimal_policy_value-baseline_value)))


def compute_mean_ratio_learning_method(learning_results, baseline_results, optimal_policy_results, mean_value_external = None):
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

    if isinstance(mean_value_external, type(None)):
        mean_value_external = mean_ratio

    sd_ratio = np.zeros((n_evaluations_per_run, ))
    for j in range(1, n_evaluations_per_run):
        sd_ratio[j] = math.sqrt( np.mean( (ratio_table[:, j] - mean_value_external[j])**2  ) )

    return mean_ratio, sd_ratio

optimized_hyperparameters = False

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/learning_methods_comparison/'
print(folder)

if optimized_hyperparameters:
    folder += 'optimized_hyperparameters/'
else:
    folder += 'non_optimized_hyperparameters/'
 

discounted_rewards_n_step_q_learning = pk.load(open(folder + 'discounted_rewards_n_step_q_learning.dat', 'rb'))
discounted_rewards_threshold_q_learning = pk.load(open(folder + 'discounted_rewards_threshold_q_learning.dat', 'rb'))
discounted_rewards_stairway_q_learning = pk.load(open(folder + 'discounted_rewards_stairway_q_learning.dat', 'rb'))
discounted_rewards_reinforce = pk.load(open(folder + 'discounted_rewards_reinforce.dat', 'rb'))
baseline_values = pk.load(open(folder + 'baseline_values.dat', 'rb'))
discounted_reward_optimal_policy = pk.load(open(folder + 'discounted_reward_optimal_policy.dat', 'rb'))




ratio_n_step_q_learning, sd_n_step_q_learning = compute_mean_ratio_learning_method(discounted_rewards_n_step_q_learning, baseline_values, discounted_reward_optimal_policy)
print(ratio_n_step_q_learning)
print(sd_n_step_q_learning)
print('-------------------------------------------')

ratio_threshold_q_learning, sd_threshold_q_learning = compute_mean_ratio_learning_method(discounted_rewards_threshold_q_learning, baseline_values, discounted_reward_optimal_policy)
print(ratio_threshold_q_learning)
print(sd_threshold_q_learning)
print('-------------------------------------------')

ratio_stairway_q_learning, sd_stairway_q_learning = compute_mean_ratio_learning_method(discounted_rewards_stairway_q_learning, baseline_values, discounted_reward_optimal_policy)
print(ratio_stairway_q_learning)
print(sd_stairway_q_learning)
print('-------------------------------------------')

# ratio_reinforce = pk.load(open(folder + 'ratio_reinforce.dat', 'rb'))
ratio_reinforce, sd_reinforce = compute_mean_ratio_learning_method(discounted_rewards_reinforce, baseline_values, discounted_reward_optimal_policy)
print(ratio_reinforce)
print(sd_reinforce)




legend = ['Q-learning', 'Threshold Q-learning', 'Stairway Q-learning', 'Reinforce']


# print(ratio_n_step_q_learning)
# print(ratio_threshold_q_learning)
# print(ratio_stairway_q_learning)
# print(ratio_reinforce)

# pk.dump(ratio_reinforce, open(folder + 'ratio_reinforce.dat', 'wb'))

print('\\begin{tabular}{c|c|c|c|c}')
print('\\textbf{Algorithm} & \\textbf{250 it.} & \\textbf{1000 it.} & \\textbf{3000 it.}\\\\')
print('\\hline')
print('QL& $' + str('%.3f'%(ratio_n_step_q_learning[1])) +'\\pm ' + str('%.3f'%(sd_n_step_q_learning[1])) + '$& $' + str('%.3f'%(ratio_n_step_q_learning[4])) +'\\pm ' + str('%.3f'%(sd_n_step_q_learning[4])) +'$&  $' +str('%.3f'%(ratio_n_step_q_learning[-1])) +'\\pm ' + str('%.3f'%(sd_n_step_q_learning[-1])) +'$\\\\')
print('TQL& $' + str('%.3f'%(ratio_threshold_q_learning[1])) +'\\pm ' + str('%.3f'%(sd_threshold_q_learning[1])) + '$& $' + str('%.3f'%(ratio_threshold_q_learning[4])) +'\\pm ' + str('%.3f'%(sd_threshold_q_learning[4])) +'$&  $' +str('%.3f'%(ratio_threshold_q_learning[-1])) +'\\pm ' + str('%.3f'%(sd_threshold_q_learning[-1])) +'$\\\\')
print('SQL& $' + str('%.3f'%(ratio_stairway_q_learning[1])) +'\\pm ' + str('%.3f'%(sd_stairway_q_learning[1])) + '$& $' + str('%.3f'%(ratio_stairway_q_learning[4])) +'\\pm ' + str('%.3f'%(sd_stairway_q_learning[4])) +'$&  $' +str('%.3f'%(ratio_stairway_q_learning[-1])) +'\\pm ' + str('%.3f'%(sd_stairway_q_learning[-1])) +'$\\\\')
print('Reinforce& $' + str('%.3f'%(ratio_reinforce[1])) +'\\pm ' + str('%.3f'%(sd_reinforce[1])) + '$& $' + str('%.3f'%(ratio_reinforce[4])) +'\\pm ' + str('%.3f'%(sd_reinforce[4])) +'$&  $' +str('%.3f'%(ratio_reinforce[-1])) +'\\pm ' + str('%.3f'%(sd_reinforce[-1])) +'$\\\\')
print('\\end{tabular}')


plt.plot(ratio_n_step_q_learning)
plt.plot(ratio_threshold_q_learning)
plt.plot(ratio_stairway_q_learning)
plt.plot(ratio_reinforce)
plt.legend(legend)
plt.ylim([0, 1])
plt.xlabel('Episodes')
plt.xticks(np.arange(0, len(ratio_threshold_q_learning), 2), np.arange(0, 3000 +1, 500)) 
plt.savefig(folder + 'comparison_learning_methods.png')
plt.show()



