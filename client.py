import connection as cn 
import numpy as np
import tqdm, math, time
import matplotlib.pyplot as plt 
import os

# create socket to connect with the simulation
sock = cn.connect(2037)

# possible actions for each state
actions = ["left", "right", "jump"]

# Hyper-parameters:
# alpha [0,1]: learning rate -> how much the agent learns from new actions
# gamma [0,1]: discount factor -> how much the agent takes future rewards into account
# epsilon [0,1]: exploration probability -> probability of choosing a randon action

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 230

# write q-table in resultado.txt file
def save_results():

    mat = np.matrix(Q)
    with open('results/resultado.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

# Q-table: 96 states X 3 actions
# 24 platforms * 4 directions = 96 states

try:
    # load pre-trained Q-table
    
    Q = np.loadtxt('results/resultado.txt')
except:
    # random Q-table 96 states X 3 actions
    # 24 platforms * 4 directions = 96 states 
    
    Q = np.random.randint(10, size=(96, 3))  

# select next action using epsilon-greedy aproach
def select_action(state):
    if np.random.random() < epsilon: # random action
        return np.random.randint(3) # [0,3)
    else:
        return(np.argmax(Q[state]))

# check if the state is terminal
def is_terminal(reward):
    return (reward == 300 or reward == -100)

# plot reward per episode X episodes
def plot_learning(episodes, reward_per_episode):
    plt.plot(range(1, episodes + 1), reward_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Reward per episode')
    plt.savefig('results/reward_per_episode.png')

def main():

    # create progress bar
    progress_bar = tqdm.tqdm(total=num_episodes)
    reward_per_episode = []

    for i in range(1, num_episodes + 1): # for each episode
        state, reward = cn.get_state_reward(sock, 'left')

        # convert state from binary to int
        state = int(state, 2)
        episode_reward = 0
        # for each step while the state is not terminal
        while True: 
            idx_act = select_action(state)
            next_state, reward = cn.get_state_reward(sock, actions[idx_act])
            reward = int(reward)
            episode_reward += reward

            # convert state from binary to int
            next_state = int(next_state, 2)

            # update q-value
            Q[state][idx_act] += alpha*( reward + gamma*max(Q[next_state]) - Q[state][idx_act])
            
            state = next_state

            if(is_terminal(reward)):
                break

        reward_per_episode.append(episode_reward)
        
        # continually update resultado.txt after 1000 episodes
        if( i % 100 == 0):
            save_results()

            # clear terminal
            os.system('cls' if os.name == 'nt' else 'clear')

            print(f"Results updated for {i} episodes")

            # plot and save graph
            plot_learning(i, reward_per_episode)
        
        # update progress bar
        progress_bar.update(1)

if __name__=="__main__":
    start_time = time.time()
    main()
    minutes = math.floor((time.time() - start_time)/60)
    seconds = (time.time() - start_time)%60
    print(f"Total training time: {minutes} minutes and {seconds:.2f} seconds")
