import os
import numpy as np
import sys

sys.path.append(os.getcwd())

import connection as cn 

# create socket to connect with the simulation
sock = cn.connect(2037)

# load Q-table from previous trainig
Q = np.loadtxt('results/resultado.txt')

# possible actions for each state
actions = ["left", "right", "jump"]

# select action with maximum Q-value
def select_action(state):
        return(np.argmax(Q[state, :]))

def is_terminal(reward):
    return (reward == 300 or reward == -100)

def main():

    for _ in range(20):
        state = 0
        while True: 
            idx_act = select_action(state)
            next_state, reward = cn.get_state_reward(sock, actions[idx_act])

            # convert state from binary to int
            next_state = int(next_state, 2)
            
            state = next_state

            if(is_terminal(reward)):
                break
        
if __name__=="__main__":
    main()