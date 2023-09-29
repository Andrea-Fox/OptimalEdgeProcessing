import pickle as pk
import matplotlib.pyplot as plt 
import numpy as np
import math
import os
import sys



def compute_ratio(learning_value, baseline_value, optimal_policy_value):
    return min(1, max(0, (learning_value-baseline_value)/(optimal_policy_value-baseline_value)))

def mean_sd_ratio_SA_method(learning_results, baseline_results, optimal_policy_results):
    n_runs = len(learning_results)
    # check if the lengths are all the same
    # if len(learning_results) != len(baseline_results) or len(learning_results)!= len(optimal_policy_results) or len(baseline_results) != len(optimal_policy_results):
    #     ValueError('Different vectors lengths')
    
    
    ratio_vector = np.zeros((n_runs, )) 

    for i in range(n_runs):
        ratio_vector[i] = compute_ratio(learning_results[i], baseline_results[i], optimal_policy_results[i])

    mean_ratio = np.mean(ratio_vector)  

    sd_ratio = np.std(ratio_vector)

    return mean_ratio, sd_ratio, ratio_vector


def mean_list_fo_list(list_of_list):
    sum_elements = 0
    n_elements = 0
    for i in range(len(list_of_list)):
        for j in range(len(list_of_list[i])):
            if list_of_list[i][j]>0:
                sum_elements += list_of_list[i][j]
                n_elements +=1
    return sum_elements/n_elements


n_devices = 3
random_environments = True

folder = os.path.dirname(os.path.abspath(sys.argv[0]))
folder += '/results/stochastic_approximation_methods_comparison/fixed_N/'
if random_environments:
    folder += str(n_devices) + '_devices/random_environments/'
else:
    folder +=  str(n_devices) + '_devices/fixed_environments/'
if not os.path.exists(folder):
    os.makedirs(folder)

list_baseline = pk.load(open(folder + 'list_baseline.dat', 'rb'))
list_rewards_naive_random_search = pk.load(open(folder + 'list_rewards_naive_random_search.dat', 'rb'))
list_rewards_enhanced_random_search = pk.load(open(folder + 'list_rewards_enhanced_random_search.dat', 'rb'))
list_rewards_sequential_SPSA = pk.load(open(folder + 'list_rewards_sequential_SPSA.dat', 'rb'))
list_rewards_SPSA_algorithm = pk.load(open(folder + 'list_rewards_SPSA_algorithm.dat', 'rb'))
list_rewards_sequential_SPSA_exact = pk.load(open(folder + 'list_rewards_sequential_SPSA_exact.dat', 'rb'))

print(len(list_rewards_enhanced_random_search))
print(list_baseline)
print(list_rewards_enhanced_random_search)


# list_rewards_naive_random_search_old = pk.load(open(folder + 'list_rewards_naive_random_search_new.dat', 'rb'))
# list_rewards_enhanced_random_search_old = pk.load(open(folder + 'list_rewards_enhanced_random_search_new.dat', 'rb'))
# list_rewards_sequential_SPSA_old = pk.load(open(folder + 'list_rewards_sequential_SPSA_new.dat', 'rb'))
# list_rewards_SPSA_algorithm_old = pk.load(open(folder + 'list_rewards_SPSA_algorithm_new.dat', 'rb'))
# list_rewards_sequential_SPSA_exact_old = pk.load(open(folder + 'list_rewards_sequential_SPSA_exact_new.dat', 'rb'))

# for each method, we create a list containing the max of each run and the lingth of the list (= n attempts)

# for each method and for each iteration, we need the max value and the number of iterations

### SEQUENTIAL SPSA EXACT
list_max_reward_sequential_SPSA_exact = []
list_n_policy_learning_sequential_SPSA_exact = []

for i in range(len(list_rewards_sequential_SPSA_exact)):
    list_max_reward_sequential_SPSA_exact.append(max(list_rewards_sequential_SPSA_exact[i]))
    list_n_policy_learning_sequential_SPSA_exact.append(len(list_rewards_sequential_SPSA_exact[i]))

###   NAIVE RANDOM SEARCH
list_max_reward_naive_random_search = []
list_n_policy_learning_naive_random_search = []

for i in range(len(list_rewards_naive_random_search)):
    list_max_reward_naive_random_search.append(max(list_rewards_naive_random_search[i]))
    list_n_policy_learning_naive_random_search.append(len(list_rewards_naive_random_search[i]))

ratio_naive_random_search_mean, ratio_naive_random_search_sd, ratio_vector_naive_random_search = mean_sd_ratio_SA_method(list_max_reward_naive_random_search, list_baseline, list_max_reward_sequential_SPSA_exact)
print('Naive random search')
print(str(ratio_naive_random_search_mean) + ' ± ' +str(ratio_naive_random_search_sd))
print(str(np.mean(list_n_policy_learning_naive_random_search)) + ' ± ' +str(np.std(list_n_policy_learning_naive_random_search)))
print('--------------------------------------')

### ENHANCED RANDOM SEARCH
list_max_reward_enhanced_random_search = []
list_n_policy_learning_enhanced_random_search = []

for i in range(len(list_rewards_enhanced_random_search)):
    list_max_reward_enhanced_random_search.append(max(list_rewards_enhanced_random_search[i]))
    list_n_policy_learning_enhanced_random_search.append(len(list_rewards_enhanced_random_search[i]))
ratio_enhanced_random_search_mean, ratio_enhanced_random_search_sd, ratio_vector_enhanced_random_search = mean_sd_ratio_SA_method(list_max_reward_enhanced_random_search, list_baseline, list_max_reward_sequential_SPSA_exact)
print('Enhanced random search')
print(str(ratio_enhanced_random_search_mean) + ' ± ' +str(ratio_enhanced_random_search_sd))
print(str(np.mean(list_n_policy_learning_enhanced_random_search)) + ' ± ' +str(np.std(list_n_policy_learning_enhanced_random_search)))
print('--------------------------------------')


### SEQUENTIAL SPSA
list_max_reward_sequential_SPSA = []
list_n_policy_learning_sequential_SPSA = []

for i in range(len(list_rewards_sequential_SPSA)):
    list_max_reward_sequential_SPSA.append(max(list_rewards_sequential_SPSA[i]))
    list_n_policy_learning_sequential_SPSA.append(len(list_rewards_sequential_SPSA[i]))
ratio_sequential_SPSA_mean, ratio_sequential_SPSA_sd, ratio_vector_sequential_SPSA = mean_sd_ratio_SA_method(list_max_reward_sequential_SPSA, list_baseline, list_max_reward_sequential_SPSA_exact)
print('Sequential SPSA')
print(str(ratio_sequential_SPSA_mean) + ' ± ' +str(ratio_sequential_SPSA_sd))
print(str(np.mean(list_n_policy_learning_sequential_SPSA)) + ' ± ' +str(np.std(list_n_policy_learning_sequential_SPSA)))
print('--------------------------------------')

### SPSA algorithm
list_max_reward_SPSA_algorithm = []
list_n_policy_learning_SPSA_algorithm = []

for i in range(len(list_rewards_SPSA_algorithm)):
    list_max_reward_SPSA_algorithm.append(max(list_rewards_SPSA_algorithm[i]))
    list_n_policy_learning_SPSA_algorithm.append(len(list_rewards_SPSA_algorithm[i]))
ratio_SPSA_algorithm_mean, ratio_SPSA_algorithm_sd, ratio_vector_SPSA_algorithm = mean_sd_ratio_SA_method(list_max_reward_SPSA_algorithm, list_baseline, list_max_reward_sequential_SPSA_exact)
print('SPSA algorithm')
print(str(ratio_SPSA_algorithm_mean) + ' ± ' +str(ratio_SPSA_algorithm_sd))
print(str(np.mean(list_n_policy_learning_SPSA_algorithm)) + ' ± ' +str(np.std(list_n_policy_learning_SPSA_algorithm)))


legend = ['Naive random \nsearch', 'Enhanced random \nsearch', 'APPI', 'SPSA']

scatter_plot = True
if scatter_plot:
    size_mean_point = 100
    plt.scatter(np.mean(list_n_policy_learning_naive_random_search), ratio_naive_random_search_mean, color = 'b', s = size_mean_point)
    plt.scatter(np.mean(list_n_policy_learning_enhanced_random_search), ratio_enhanced_random_search_mean, color = 'r', s = size_mean_point)
    plt.scatter(np.mean(list_n_policy_learning_sequential_SPSA), ratio_sequential_SPSA_mean, color = 'g', s = size_mean_point)
    plt.scatter(np.mean(list_n_policy_learning_SPSA_algorithm), ratio_SPSA_algorithm_mean, color = 'c', s = size_mean_point)

    alpha_points = 0.25
    plt.scatter(list_n_policy_learning_naive_random_search, ratio_vector_naive_random_search, color = 'b', alpha = alpha_points)
    plt.scatter(list_n_policy_learning_enhanced_random_search, ratio_vector_enhanced_random_search, color = 'r', alpha = alpha_points)
    plt.scatter(list_n_policy_learning_sequential_SPSA, ratio_vector_sequential_SPSA, color = 'g', alpha = alpha_points)
    plt.scatter(list_n_policy_learning_SPSA_algorithm, ratio_vector_SPSA_algorithm, color = 'c', alpha = alpha_points)

    plt.xticks(np.arange(0, 21, 2))
    # plt.hlines(y = max_exact_reward, xmin = 0, xmax = 20)

    # plt.yticks(ticks=np.arange(baseline, max_exact_reward + (max_exact_reward - baseline)/10, (max_exact_reward + (max_exact_reward - baseline)/10 - baseline)/11),labels =  np.arange(0, 1.1, .1))
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.legend(legend)
    plt.savefig(folder + 'comparison_scatterplot.png')
    plt.show()
 
boxplot = True
if boxplot:
    plt.boxplot([ratio_vector_naive_random_search, ratio_vector_enhanced_random_search, ratio_vector_sequential_SPSA, ratio_vector_SPSA_algorithm])
    # plt.yticks(ticks=np.arange(baseline, max_exact_reward + (max_exact_reward - baseline)/10, (max_exact_reward + (max_exact_reward - baseline)/10 - baseline)/11),labels =  np.arange(0, 1.1, .1))
    # plt.ylim([baseline, max_exact_reward])
    plt.title('Ratio rewards')
    plt.xticks(ticks=np.arange(1, 5), labels=legend)
    plt.savefig(folder + 'comparison_ratio_boxplot.png')
    plt.show()

    plt.boxplot([list_n_policy_learning_naive_random_search, list_n_policy_learning_enhanced_random_search, list_n_policy_learning_sequential_SPSA, list_n_policy_learning_SPSA_algorithm])
    plt.yticks(ticks=np.arange(0, 21, 2))
    plt.title('Policy learning operations')
    plt.ylim([0, 20])
    plt.xticks(ticks=np.arange(1, 5), labels=legend)
    plt.savefig(folder + 'comparison_n_policy_learning_boxplot.png')
    plt.show()