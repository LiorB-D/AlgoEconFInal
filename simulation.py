import sys
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




def train_agents(train_interval, target_update_interval, test_file):
    with open(test_file, 'rb') as f:
        adjacency_matrices = np.load(f)
    count = 0
    QRL_cutter = QHandler()
    QRL_fixer = QHandler()
    for adj_m in adjacency_matrices:
        count = count + 1
        simulate(adj_m, QRL_fixer, QRL_cutter)
        if count > 1:
            if count % train_interval == 0:
                QRL_fixer.trainQNetwork()
                QRL_cutter.trainQNetwork()
            if count % target_update_interval == 0:
                QRL_fixer.updateTargetNetwork()
                QRL_cutter.updateTargetNetwork()




if __name__ == '__main__':
    # n = int(input())
    # adj_matrix = np.array([list(map(int, input().split())) for _ in range(n)])
    # print(adj_matrix)
    # simulate(adj_matrix)
    if (len(sys.argv) != 2):
        print("Usage: python3 simulation.py train/run <test_file>")
        exit(1)
    
    if sys.argv[1] == "train":
        train_agents(1, 2, sys.argv[1])
