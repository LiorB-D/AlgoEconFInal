import numpy as np
from shannon_switching_game import ShannonSwitchingGame, Player
import unittest

class TestShannonSwitchingGame(unittest.TestCase):

    def test_get_winner(self):
        adj = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        game = ShannonSwitchingGame(adj, 0, 1)
        assert game.get_winner() == Player.CUTTER

        game.adj_matrix = np.array([
            [0, -1, 1],
            [-1, 0, 1],
            [1, 1, 0]
        ])
        assert game.get_winner() == Player.FIXER

        game.adj_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        assert game.get_winner() == None



if __name__ == '__main__':
    unittest.main()
