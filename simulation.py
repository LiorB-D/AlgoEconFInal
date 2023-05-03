import numpy as np

from RLAgent import QHandler
from shannon_switching_game import Player, ShannonSwitchingGame

CUTTER_WIN = -1, 1
FIXER_WIN = 1, -1

# take adjacency matrix as input
def simulate(adj_matrix: np.ndarray) -> None: 
    SSG = ShannonSwitchingGame(adj_matrix)
    QRL_cutter = QHandler(SSG)
    QRL_fixer = QHandler(SSG)

    reward = 0, 0
    while reward == (0, 0):
        if SSG.player_turn == Player.CUTTER:
            QRL = QRL_cutter
        else:
            QRL = QRL_fixer
        move, _ = QRL.get_best_move()
        print(move)
        reward = SSG.take_action(move)
    
    if reward == CUTTER_WIN:
        print("CUTTER wins!")
    elif reward == FIXER_WIN:
        print("FIXER wins!")
    else:
        print("ERROR: invalid termination of game!")
        exit(1)

if __name__ == '__main__':
    n = int(input())
    adj_matrix = np.array([list(map(int, input().split())) for _ in range(n)])
    print(adj_matrix)
    simulate(adj_matrix)
