import connection as cn 

sock = cn.connect(2037)
actions = ["left", "right", "jump"]

while True:
    idx_act = int(input("Próxima ação: "))
    estado, recompensa = cn.get_state_reward(sock, actions[idx_act])
   
    print(f"estado em binário = {estado}, estado em decimal = {int(estado, 2)},recompensa = {recompensa}")
