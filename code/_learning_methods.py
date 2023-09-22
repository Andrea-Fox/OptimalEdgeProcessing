import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd 
import scipy
import math
import random

from scipy.special import softmax




def egreedy_policy(q_values, state, epsilon):
    if state >= num_states/2:
        return 1
    if random.random() < epsilon:
        return random.randrange(2)
    else:
        return np.argmax(q_values[state, :])
    

def boltzmann_exploration(q_values, state, exploration_rate, state_occurrences):
    # we choose C_t equal to the maximum between the values
    num_states = q_values.shape[0]
    
    # in the system studied the only possible action when server is available is 1
    if state >= num_states/2:
        return 1, state_occurrences, [0, 1]

    if np.sum(state_occurrences) == 0:
        return random.randrange(2)
    else:
        C = exploration_rate*max(abs(q_values[state, 0] - q_values[state, 1]), 0.1)
        if C==0:
            C = 1

        probabilities_vector = softmax([(math.log(state_occurrences)/C) *  q_values[state, i]      for i in range(2)])
        # now we need to draw an action according to the probabilty given by probabilites_vector

        action = random.choices(np.arange(0, 2), weights = probabilities_vector)[0]

        return action


def compute_average_reward(env, policy):
    # n_steps has to be such that gamma**n_steps < 0.001
    min_future_weight = 0.001
    n_simulations = 250
    average_rewards = []
    if len(policy.shape) == 1 or policy.shape[1] == 1:
        # we have a vector indicating the optimal policy
        for _ in range(n_simulations):
            total_reward = 0
            # state = env.compute_index(1, random.randrange(env.B+1), random.choices([0, 1]))
            state = env.compute_index(1, random.randrange(env.B+1), sum(random.choices([0, 1], weights=[1-env.p_z, env.p_z], k = 1 ))) 
            i = 0
            while env.gamma**i > min_future_weight:
                action = policy[state]
                if action != 0 and action != 1:
                    print(action)
                next_state, reward, _ = env.step_stochastic_h(state, action)
                total_reward += (env.gamma**i) * reward
                state = next_state
                i+= 1
            average_rewards.append(total_reward)
    else:
        # we have a the Q-values matrix
        for _ in range(n_simulations):
            total_reward = 0
            state = env.compute_index(1, random.randrange(env.B+1), sum(random.choices([0, 1], weights=[1-env.p_z, env.p_z], k = 1 ))) 
            i = 0
            while env.gamma**i > min_future_weight:
                action = np.argmax(policy[state, :])
                next_state, reward, _ = env.step_stochastic_h(state, action)
                total_reward += (env.gamma)**(i) * reward
                state = next_state
                i += 1
            average_rewards.append(total_reward)
    # print(average_rewards)
    return np.mean(average_rewards)

def average_reward_regenerative_model(env, policy):
    n_simulations = 50000
    max_steps = 100
    n_steps_total = 0
    average_rewards = []
    if len(policy.shape) == 1:
        # we have the optimal solution
        for _ in range(n_simulations):
            total_reward = 0
            state = env.compute_index(1, random.randrange(env.B+1), sum(random.choices([0, 1], weights=[1-env.p_z, env.p_z], k = 1 ))) 
            n_steps = 0
            done = False
            while not done:
                action = policy[state]
                next_state, reward, _ = env.step_stochastic_h(state, action)
                total_reward += env.gamma**n_steps * reward
                n_steps += 1
                n_steps_total += 1
                # print(env.compute_state_values(next_state), reward)
                if env.compute_state_values(next_state)[0] == 1 or n_steps > max_steps:
                    done = True
                state = next_state
            average_rewards.append(total_reward)
    else:
        # we are computing the values of the optimal policy (we have the proper reward)
        for _ in range(n_simulations):
            total_reward = 0
            state = env.compute_index(1, random.randrange(env.B+1), sum(random.choices([0, 1], weights=[1-env.p_z, env.p_z], k = 1 ))) 
            n_steps = 0
            done = False
            while not done:
                action = np.argmax(policy[state, :])
                next_state, reward, _ = env.step_stochastic_h(state, action)
                total_reward += env.gamma**n_steps * reward
                n_steps += 1
                n_steps_total += 1
                # print(env.compute_state_values(next_state), reward)
                if env.compute_state_values(next_state)[0] == 1 or n_steps > 100:
                    done = True
                state = next_state
            average_rewards.append(total_reward)
    # return sum(average_rewards)/n_steps_total
    return sum(average_rewards)/n_simulations, n_steps_total/n_simulations      # first term is E[C], second term is E[l]



def threshold_q_learning(env, value_iteration_policy=None ,num_episodes=500, learning_rate=0.5, discount_factor=0.9 , n = 1, beta = 0.75, only_last_measurement = False): 

    num_states = env.M * (env.B+1) * 2
    num_actions = 2
    q_values = np.zeros((num_states, num_actions))
    for i in range(env.compute_index(env.M, env.B, 0), env.num_states):
        q_values[i, 1] = 0.001
    ep_rewards = []

    discounted_reward_vector = []

    max_length_episode = env.M*2

    state_counts = np.zeros((num_states, 1))
    
    best_estimate = - math.inf      
    best_q_values = np.zeros((num_states, num_actions)) 
    
    if not isinstance(value_iteration_policy, type(None)):
        discounted_reward_optimal_policy = discounted_reward(env, value_iteration_policy)
    else: 
        discounted_reward_optimal_policy =-1    # print(average_reward_optimal_policy)
    
    for episode_index in range(num_episodes):    
        if (episode_index % 250 == 0 and not only_last_measurement):
            discounted_reward_q_values = compute_average_reward(env, q_values)
            discounted_reward_vector.append(discounted_reward_q_values)
            if discounted_reward_q_values > best_estimate:
                for index in range(best_q_values.shape[0]):
                    best_q_values[index, 0] = q_values[index, 0]
                    best_q_values[index, 1] = q_values[index, 1]
                best_estimate = discounted_reward_q_values

        if only_last_measurement and (episode_index == num_episodes-1 ) :
            best_estimate = compute_average_reward(env, q_values)
            for index in range(best_q_values.shape[0]):
                best_q_values[index, 0] = q_values[index, 0]
                best_q_values[index, 1] = q_values[index, 1]
        
        future_states = []
        future_rewards = []
        future_actions = []

        t = 0
        tau = t - n + 1
        terminal_time = math.inf
        
        state = env.reset_all_states()
        # we add S_0 to the list of states
        future_states.append(state)   
        
        # we add R_0 = 0 to the list of rewards, as the rewarda have to be considered only from 1 onwards
        future_rewards.append(-1)
        
        done = False
        reward_sum = 0         

        while tau < terminal_time - 1:   

            if (t < terminal_time):
                # find the optimal action according to the current policy
                aoi, energy, server = env.compute_state_values(state)
                if server == 1 or (aoi == env.M and energy == env.B):
                    action = 1
                elif server == 0:
                    action = boltzmann_exploration(q_values, state=state, exploration_rate = 1.75, state_occurrences=state_counts[state])
                future_actions.append(action)

                # take action A_t
                state_counts[state] += 1
                next_state, reward, done = env.step_stochastic_h(state, action)

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)    
                
                # If S_{t+1} is terminal, then update terminal_time
                if done or t > max_length_episode:
                    terminal_time = t + 1
                            
            tau = t - n + 1

            # fare in modo che al tempo tau + n (o terminal time,  ma in quel caso adeguare tutto) si scelga 
            # il valore dato dalla stima corrente 
            # Va fatto in modo che io abbia n step finche' posso e poi averne fino alla fine, sfruttando tutte
            # le osservazioni a disposizione
            
            if tau >= 0:
                final_index = min(terminal_time, tau + n)
                if tau + n <= final_index:
                    # we have the time to consider all the n steps
                    gamma_list = [discount_factor**x for x in list(range(0, n))]
                    final_return = np.dot(gamma_list, future_rewards[(tau+1):(tau+n + 1)])
                    final_return  += (discount_factor ** n) * np.max(q_values[ future_states[tau + n] ] )
                else: # terminal time > tau + n
                    gamma_list = [discount_factor**x for x in list(range(0, terminal_time -1 - tau))]
                    final_return = np.dot(gamma_list, future_rewards[(tau+1):(terminal_time )])
                    final_return  += (discount_factor ** (terminal_time - 1- tau)) * np.max(q_values[ future_states[terminal_time] ] )

                q_values[ future_states[tau],  future_actions[tau] ] += (learning_rate/(state_counts[future_states[tau]])**beta) * (final_return - q_values[ future_states[tau],  future_actions[tau] ] )

                # we should check if the estimate has increased. If so, then look at values on the left and change them accordingly; 
                # otherwise adapt values on the right
                # update of q_values table according to the structure of the solution
                update_state_aoi, update_state_energy, update_state_server_availability = env.compute_state_values(future_states[tau])
                # we have to update all values on the left and all values on the right accordingly
                # update of values on the right

                #if update_state_aoi < env.M:
                for aoi_to_update in range(update_state_aoi+1, env.M+1):
                    state_to_update = env.compute_index(aoi_to_update, update_state_energy, update_state_server_availability)
                    q_values[state_to_update, future_actions[tau]] = min(q_values[future_states[tau], future_actions[tau]], q_values[state_to_update, future_actions[tau]])
                # if update_state_aoi > 0:
                for aoi_to_update in range(1, update_state_aoi):
                    state_to_update = env.compute_index(aoi_to_update, update_state_energy, update_state_server_availability)
                    q_values[state_to_update, future_actions[tau]] = max(q_values[future_states[tau], future_actions[tau]], q_values[state_to_update, future_actions[tau]])


            state = next_state
            t += 1
        
        # best_q_values = update_all_q_values(best_q_values)
        ep_rewards.append(reward_sum) 
    
    # print(state_counts)
    # print(q_values - best_q_values)
    return best_q_values, best_estimate, discounted_reward_vector


def n_step_q_learning_comparison(env, value_iteration_policy = None, num_episodes=500, learning_rate=0.5, discount_factor=0.9, n = 1, beta = 0.75, only_last_measurement = False): 

    num_states = env.M * (env.B+1) * 2
    num_actions = 2
    q_values = np.zeros((num_states, num_actions))
    for i in range(env.compute_index(env.M, env.B, 0), env.num_states):
        q_values[i, 1] = 0.001
    ep_rewards = []

    discounted_reward_vector = []

    max_length_episode = env.M*2

    state_counts = np.zeros((num_states, 1))
    
        # steps we use backwards to find good estimates
    best_estimate = - math.inf      
    best_q_values = np.zeros((num_states, num_actions)) 
    
    if not isinstance(value_iteration_policy, type(None)):
        discounted_reward_optimal_policy = discounted_reward(env, value_iteration_policy)
    else: 
        discounted_reward_optimal_policy =-1
            # print(average_reward_optimal_policy)
    for episode_index in range(num_episodes):
        if (episode_index % 250 == 0 and not only_last_measurement):
            discounted_reward_q_values = compute_average_reward(env, q_values)
            discounted_reward_vector.append(discounted_reward_q_values)
            if discounted_reward_q_values > best_estimate:
                for index in range(best_q_values.shape[0]):
                    best_q_values[index, 0] = q_values[index, 0]
                    best_q_values[index, 1] = q_values[index, 1]
                best_estimate = discounted_reward_q_values

        if only_last_measurement and (episode_index == num_episodes-1 ) :
            best_estimate = compute_average_reward(env, q_values)
            for index in range(best_q_values.shape[0]):
                    best_q_values[index, 0] = q_values[index, 0]
                    best_q_values[index, 1] = q_values[index, 1]

        future_states = []
        future_rewards = []
        future_actions = []

        t = 0
        tau = t - n + 1
        terminal_time = math.inf
        

        state = env.reset_all_states()
        
        # we add S_0 to the list of states
        future_states.append(state)   
        
        # we add R_0 = 0 to the list of rewards, as the rewarda have to be considered only from 1 onwards
        future_rewards.append(-1)


        # future_actions.append(0)
        
        done = False
        reward_sum = 0
         

        while tau < terminal_time - 1:   

            if (t < terminal_time):

                # find the optimal action according to the current policy
                aoi, energy, server = env.compute_state_values(state)
                if server == 1 or (aoi == env.M and energy == env.B):
                    action = 1
                elif server == 0:
                    action = boltzmann_exploration(q_values, state=state, exploration_rate = 1.75, state_occurrences=state_counts[state])
                future_actions.append(action)

                # take action A_t
                state_counts[state] += 1
                next_state, reward, done = env.step_stochastic_h(state, action)

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)    
                
                # If S_{t+1} is terminal, then update terminal_time
                if done or t > max_length_episode:
                    terminal_time = t + 1
                            
            tau = t - n + 1

            # fare in modo che al tempo tau + n (o terminal time,  ma in quel caso adeguare tutto) si scelga 
            # il valore dato dalla stima corrente 
            # Va fatto in modo che io abbia n step finche' posso e poi averne fino alla fine, sfruttando tutte
            # le osservazioni a disposizione
            
            if tau >= 0:
                final_index = min(terminal_time, tau + n)
                if tau + n <= final_index:
                    # we have the time to consider all the n steps
                    gamma_list = [discount_factor**x for x in list(range(0, n))]
                    final_return = np.dot(gamma_list, future_rewards[(tau+1):(tau+n + 1)])
                    final_return  += (discount_factor ** n) * np.max(q_values[ future_states[tau + n] ] )
                else: # terminal time > tau + n
                    gamma_list = [discount_factor**x for x in list(range(0, terminal_time -1 - tau))]
                    final_return = np.dot(gamma_list, future_rewards[(tau+1):(terminal_time )])
                    final_return  += (discount_factor ** (terminal_time - 1- tau)) * np.max(q_values[ future_states[terminal_time] ] )

                # update of q_values table
                q_values[ future_states[tau],  future_actions[tau] ] += (learning_rate/(state_counts[future_states[tau]])**beta)  * (final_return - q_values[ future_states[tau],  future_actions[tau] ] )

            state = next_state
            t += 1
        
        ep_rewards.append(reward_sum) 
    
    return best_q_values, best_estimate, discounted_reward_vector


def stairway_q_learning(env, value_iteration_policy = None,num_episodes=3001, exploration_rate=0.1,
               learning_rate=0.5, discount_factor=0.9, n = 1, beta = 0.75, only_last_measurement = False): 

    num_states = env.M * (env.B+1) * 2
    num_actions = 2
    q_values = np.zeros((num_states, num_actions))
    for i in range(env.compute_index(env.M, env.B, 0), env.num_states):
        q_values[i, 1] = 0.001
    ep_rewards = []

    discounted_reward_vector = []


    max_length_episode = env.M*2

    state_counts = np.zeros((num_states, 1))
    
    best_estimate = - math.inf      
    best_q_values = np.zeros((num_states, num_actions)) 
    
    if not isinstance(value_iteration_policy, type(None)):
        discounted_reward_optimal_policy = discounted_reward(env, value_iteration_policy)
    else: 
        discounted_reward_optimal_policy =-1
        
    for episode_index in range(num_episodes):    
        if (episode_index % 250 == 0 and not only_last_measurement):
            discounted_reward_q_values = compute_average_reward(env, q_values)
            discounted_reward_vector.append(discounted_reward_q_values)
            if discounted_reward_q_values > best_estimate:
                for index in range(best_q_values.shape[0]):
                    best_q_values[index, 0] = q_values[index, 0]
                    best_q_values[index, 1] = q_values[index, 1]
                best_estimate = discounted_reward_q_values
            

        if only_last_measurement and (episode_index == num_episodes-1 ) :
            best_estimate = compute_average_reward(env, q_values)
            for index in range(best_q_values.shape[0]):
                best_q_values[index, 0] = q_values[index, 0]
                best_q_values[index, 1] = q_values[index, 1]
            
        future_states = []
        future_rewards = []
        future_actions = []

        t = 0
        tau = t - n + 1
        terminal_time = math.inf
        
        state = env.reset_all_states()
        # we add S_0 to the list of states
        future_states.append(state)   
        
        # we add R_0 = 0 to the list of rewards, as the rewarda have to be considered only from 1 onwards
        future_rewards.append(-1)
        
        done = False
        reward_sum = 0
         
        while tau < terminal_time - 1:   

            if (t < terminal_time):
                # find the optimal action according to the current policy
                aoi, energy, server = env.compute_state_values(state)
                if server == 1 or (aoi == env.M and energy == env.B):
                    action = 1
                elif server == 0:
                    action = boltzmann_exploration(q_values, state=state, exploration_rate = 1.75, state_occurrences=state_counts[state])
                future_actions.append(action)

                # take action A_t
                state_counts[state] += 1
                next_state, reward, done = env.step_stochastic_h(state, action)

                # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                # indeed thay are going to be element t+1 in the respective lists
                future_states.append(next_state)
                reward_sum += reward
                future_rewards.append(reward)    
                
                # If S_{t+1} is terminal, then update terminal_time
                if done or t > max_length_episode:
                    terminal_time = t + 1
                            
            tau = t - n + 1

            
            if tau >= 0:
                final_index = min(terminal_time, tau + n)
                if tau + n <= final_index:
                    # we have the time to consider all the n steps
                    gamma_list = [discount_factor**x for x in list(range(0, n))]
                    final_return = np.dot(gamma_list, future_rewards[(tau+1):(tau+n + 1)])
                    final_return  += (discount_factor ** n) * np.max(q_values[ future_states[tau + n] ] )
                else: # terminal time > tau + n
                    gamma_list = [discount_factor**x for x in list(range(0, terminal_time -1 - tau))]
                    final_return = np.dot(gamma_list, future_rewards[(tau+1):(terminal_time )])
                    final_return  += (discount_factor ** (terminal_time - 1- tau)) * np.max(q_values[ future_states[terminal_time] ] )

                q_values[ future_states[tau],  future_actions[tau] ] += (learning_rate/(state_counts[future_states[tau]]**beta)) * (final_return - q_values[ future_states[tau],  future_actions[tau] ] )

                # update of q_values table according to the structure of the solution
                update_state_aoi, update_state_energy, update_state_server_availability = env.compute_state_values(future_states[tau])
                # if update_state_aoi < env.M:
                    # states that are smaller
                for energy_to_update in range(0, update_state_energy+1):
                    for aoi_to_update in range(update_state_aoi, env.M+1):
                        state_to_update = env.compute_index(aoi_to_update, energy_to_update, update_state_server_availability)
                        q_values[state_to_update, future_actions[tau]] = min(q_values[future_states[tau], future_actions[tau]], q_values[state_to_update, future_actions[tau]])
                
                    # states that are bigger
                for energy_to_update in range(update_state_energy, env.B+1):
                    for aoi_to_update in range(1, update_state_aoi+1):
                        state_to_update = env.compute_index(aoi_to_update, energy_to_update, update_state_server_availability)
                        q_values[state_to_update, future_actions[tau]] = max(q_values[future_states[tau], future_actions[tau]], q_values[state_to_update, future_actions[tau]])
 

            state = next_state
            t += 1
        
        # best_q_values = update_all_q_values(best_q_values)
        ep_rewards.append(reward_sum) 
     
    # print(state_counts)
    # print(q_values - best_q_values)
    return best_q_values, best_estimate, discounted_reward_vector


def reinforce(env, policy_parameter = 5, learning_rate_constant = 0.01, learning_rate_exponent = 2, n_episodes = 3000):
    
    theta = env.M*np.ones((env.B+1, ))
    n_steps = 100
    n_simulations = 250
    vector_average_reward = []

    alpha = lambda x: learning_rate_constant/(x+1)**learning_rate_exponent
    
    improvement = math.inf
    
    episode_length = 100
    length_epoch = 1

    min_future_weight = 0.001
    new_theta = np.zeros((env.B+1, ))

    discounted_reward_vector = []

    for i in range(env.B+1):
        new_theta[i] = theta[i]

    for episode in range(n_episodes):
        # generate an episode
        state_list = []
        action_list = []
        reward_list = []
        state = env.reset()
        # print(env.compute_state_values(state))
        state_list.append(state)
        reward_list.append(-math.inf)
        # loop for each step of the episode
        for t in range(episode_length):
            x, e, z = env.compute_state_values(state)
            if z == 0:
                # print(x, 1/(1+math.exp(10*(theta[e]-x-0.5))))
                action = np.random.binomial(1, 1/(1+math.exp(policy_parameter*(theta[e]-x-0.5))))
                # print(action)
            else: 
                action = 1
            action_list.append(action)
            next_state, reward, _ = env.step_stochastic_h(state, action)
            reward_list.append(reward)
            state_list.append(next_state)
            state = next_state
        # print('-----------------------')
        # print(state_list)
        for t in range(episode_length):
            state = state_list[t]
            x_t, e_t, z_t = env.compute_state_values(state)
            action = action_list[t]
            G = np.dot([env.gamma**(k-t-1) for k in range(t+1, episode_length+1)], reward_list[(t+1):(episode_length+1)])
            # print(G)
            if z_t == 0: # when z_t==1, the policy is independent of the parameter theta
                if action == 0:
                    derivative = (policy_parameter*math.exp(policy_parameter*(theta[e_t] - x_t - 0.5)))/(1 + math.exp(policy_parameter*(theta[e_t] - x_t - 0.5)))
                    theta[e_t] = min(env.M, max(1, theta[e_t] - alpha(episode//length_epoch) * env.gamma**t * G * derivative))
                elif action == 1:
                    theta[e_t] = min(env.M, max(1, theta[e_t] + alpha(episode//length_epoch) * env.gamma**t * G * policy_parameter/(1 + math.exp(policy_parameter*(theta[e_t] - x_t - 0.5))) ))

            # impose order on values of theta
            for e in range(1, env.B+1):
                theta[e] = min(theta[e], math.ceil(theta[e-1])+0.5) 

        if (episode) % 250 == 0:
            # print((episode+1), theta)
            average_rewards = []
            for _ in range(n_simulations):
                total_reward = 0
                state = env.compute_index(1, random.randrange(env.B+1), sum(random.choices([0, 1], weights=[1-env.p_z, env.p_z], k = 1 ))) 
                i = 0
                while env.gamma**i > min_future_weight:
                    x, e, z = env.compute_state_values(state)
                    if z==0:
                        action = np.random.binomial(1, 1/(1+math.exp(policy_parameter*(theta[e]-x-0.5))))
                    else:
                        action = 1
                    next_state, reward, _ = env.step_stochastic_h(state, action)
                    total_reward += (env.gamma**i) * reward
                    state = next_state
                    i += 1
                average_rewards.append(total_reward)
            discounted_reward_vector.append(np.mean(average_rewards))


    best_estimate = np.max(discounted_reward_vector)
    return best_estimate, discounted_reward_vector

