import connection as cn 

sock = cn.connect(2037)
act = ["jump", "left", "right"]

while True:
    idx_act = int(input("Próxima ação: "))
    estado, recompensa = cn.get_state_reward(sock, act[idx_act])
   
    print(f"estado em binário = {estado}, estado em decimal = {int(estado, 2)},recompensa = {recompensa}")
