import connection as cn 
import numpy.random as rand

# create socket to connect with the simulation
sock = cn.connect(2037)

# possible actions for each state
actions = ["jump", "left", "right"]

# Hyper-parameters:
# alpha [0,1]: learning rate -> how much the agent learns from new actions
# gamma [0,1]: discount factor -> how much the agent takes future rewards into account
# epsilon [0,1]: exploration probability -> probability of choosing a randon action

alpha = 0.5
gamma = 0.8
epsilon = 0.05

# random Q-table 96 states X 3 actions
# 24 platforms * 4 directions = 96 states 

Q = rand.randint(10, size=(96, 3))  

def select_action(state):
    n = rand.random_sample()
    if n < epsilon:
        return rand.randint(0, 4)
    
    return(max(Q[state]))

def main():

    while True: # for each episode
        state = 0
        while True: # for each step in episode
            while True: # while state is not terminal
                idx_act = select_action(state)
                next_state, reward = cn.get_state_reward(sock, actions[idx_act])
                Q[state][idx_act] += alpha*( reward + gamma*max(Q[next_state]) - Q[state][idx_act])
                state = next_state

if __name__=="__main__":
    main()