import connection as cn 
import numpy as np
import tqdm, math, time

# create socket to connect with the simulation
sock = cn.connect(2037)

# possible actions for each state
actions = ["left", "right", "jump"]

# Hyper-parameters:
# alpha [0,1]: learning rate -> how much the agent learns from new actions
# gamma [0,1]: discount factor -> how much the agent takes future rewards into account
# epsilon [0,1]: exploration probability -> probability of choosing a randon action

alpha = 0.5
gamma = 0.9
epsilon = 0.1
num_episodes = 3000

pre_trained = true

if pre_trained:
    # load pre-trained Q-table
    
    Q = np.loadtxt('results/resultado.txt')
else:
    # random Q-table 96 states X 3 actions
    # 24 platforms * 4 directions = 96 states 
    
    Q = np.random.randint(10, size=(96, 3))  

# select next action using epsilon-greedy aproach
def select_action(state):
    if np.random.random() < epsilon: # random action
        return np.random.randint(3) # [0,3)
    else:
        return(np.argmax(Q[state, :]))

# check if the state is terminal
def is_terminal(reward):
    return (reward == 300 or reward == -100)

# write q-table in resultado.txt file
def save_results():

    mat = np.matrix(Q)
    with open('results/resultado.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

def main():

    # create progress bar
    progress_bar = tqdm.tqdm(total=num_episodes)

    for i in range(num_episodes): # for each episode
        state = 0
        # for each step while the state is not terminal
        while True: 
            idx_act = select_action(state)
            next_state, reward = cn.get_state_reward(sock, actions[idx_act])

            # convert state from binary to int
            next_state = int(next_state, 2)

            temporal_diference = max(Q[next_state]) - Q[state][idx_act]

            # update q-value
            Q[state][idx_act] += alpha*( reward + gamma*temporal_diference)
            
            state = next_state

            # continually update resultado.txt after 1000 episodes
            if( i % 1000 == 0):
                save_results()

            if(is_terminal(reward)):
                break
        
        # update progress bar
        progress_bar.update(1)
    
    save_results()

if __name__=="__main__":
    start_time = time.time()
    main()
    minutes = math.floor((time.time() - start_time)/60)
    seconds = (time.time() - start_time)%60
    print(f"Total training time: {minutes} minutes and {seconds:.2f} seconds")
