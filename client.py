import connection as cn

s = cn.connect(2037)

state, reward = cn.get_state_reward(s, "jump")
