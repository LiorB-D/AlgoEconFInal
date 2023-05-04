import numpy as np

from shannon_switching_game import ShannonSwitchingGame, Player
from RLAgent import QHandler, Experience

CUTTER_WIN = -1, 1
FIXER_WIN = 1, -1

# take adjacency matrix as input

def simulate(adj_matrix: np.ndarray, QRL_fixer, QRL_cutter) -> int: 
    SSG = ShannonSwitchingGame(adj_matrix)
    roundCount = 0
    reward = 0, 0
    while reward == (0, 0):
        if SSG.player_turn == Player.CUTTER:
            QRL = QRL_cutter
            print("Cutter is going")
        else:
            QRL = QRL_fixer
            print("Fixer is going")

        move, _ = QRL.get_best_move(SSG.adj_matrix)
        print("Move: ")
        print(move)
        reward = SSG.take_action(move)
        
        newExp = Experience(SSG.adj_matrix, move, 0)
        QRL.expReplay.append(newExp)
        if roundCount > 0:
            if QRL == QRL_cutter:
                QRL_fixer.expReplay[-1].reward = -1 * reward[0]
                QRL_cutter.expReplay[-1].reward = reward[1]
                QRL_fixer.expReplay[-1].setSPrime(SSG.adj_matrix)
            else:
                QRL_cutter.expReplay[-1].reward = -1 * reward[1]
                QRL_fixer.expReplay[-1].reward = reward[0]
                QRL_cutter.expReplay[-1].setSPrime(SSG.adj_matrix)
            
        roundCount = roundCount + 1
    if reward == CUTTER_WIN:
        print("CUTTER wins!")
    elif reward == FIXER_WIN:
        print("FIXER wins!")
    else:
        print("ERROR: invalid termination of game!")
        exit(1)




def train_agents(rounds, train_interval, target_update_interval):
    count = 0
    QRL_cutter = QHandler()
    QRL_fixer = QHandler()
    for i in range(0, rounds):
        count = count + 1
        adj_m = np.ones((20,20))
        simulate(adj_m, QRL_fixer, QRL_cutter)
        if count > 1:
            if count % train_interval == 0:
                QRL_fixer.trainQNetwork()
                QRL_cutter.trainQNetwork()
            if count % target_update_interval == 0:
                QRL_fixer.updateTargetNetwork()
                QRL_cutter.updateTargetNetwork()




if __name__ == '__main__':
    n = int(input())
    adj_matrix = np.array([list(map(int, input().split())) for _ in range(n)])
    print(adj_matrix)
    simulate(adj_matrix)
