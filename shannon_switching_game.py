from enum import Enum

import numpy as np


class Player(Enum): 
    FIXER = 1
    CUTTER = 2

### -1 = Reinforced Edge
### 0 = No Edge
### 1 = Normal Edge


class ShannonSwitchingGame:

    def __init__(self, adj_matrix : np.ndarray, source: int, target: int) -> None:   
        """
        Initializes the ShannonSwitchingGame instance.

        Args:
        - adj_matrix: An adjacency matrix that describes the graph of the game.
        - source: The index of the source node.
        - target: The index of the target node.
        """
        n, m = adj_matrix.shape
        assert np.all(np.logical_or(adj_matrix == 0, adj_matrix == 1)), "Graph must only contain 0s or 1s to begin"
        assert n == m, "Adjacency matrix must be square"
        assert n <= 20, "Adjacency matrix must be at most 20x20"
        assert m <= 20, "Adjacency matrix must be at most 20x20"
        assert source < n, "Source node must be in the graph"
        assert target < n, "Target node must be in the graph"
        assert len(adj_matrix.shape) == 2, "Adjacency matrix must be 2D"

        pad_rows = max(20 - n, 0)
        pad_cols = max(20 - m, 0)

        # Apply the padding
        adj_matrix = np.pad(adj_matrix, ((0, pad_rows), (0, pad_cols)), 'constant', constant_values=0)


        self.adj_matrix = np.triu(adj_matrix, k=1).astype(np.int8)
        self.source : int = 0
        self.target : int = 0
        self.player_turn : Player = Player.CUTTER


    def valid_moves(self) -> np.ndarray:
        """
        Returns an array of valid moves for the current player.

        Returns:
        - An array of valid moves for the current player.
        """
        return np.argwhere(self.adj_matrix == 1)
    

    def get_winner(self) -> Player | None:
        """
        Returns the winner of the game, if there is one.

        Returns:
        - The Player who won the game, or None if no one has won yet.
        """
        #TODO: Implement
        return None
    

    def __cut(self, n1: int, n2: int) -> tuple[int, int]:
        """
        Cuts the edge between nodes n1 and n2 and returns rewards.

        Args:
            n1 (int): The index of the first node. Must be less than n2.
            n2 (int): The index of the second node. Must be greater than n1.
            
        Returns:
            A tuple of two integers representing rewards:
            - If the move leads to CUTTER winning, (-1, 1)
            - If the move leads to neither player winning, (0, 0)
            - If the move is invalid, (0, -1)
        """
        if self.adj_matrix[n1, n2] == 1:
            self.adj_matrix[n1, n2] = 0
            winner = self.get_winner()
            if winner == Player.CUTTER:
                return -1, 1
            return 0, 0
        else: # invalid move case
            return 0, -1
        
    def __fix(self, n1: int, n2: int) -> tuple[int, int]:
        """
        Fixes the edge between nodes n1 and n2 and returns rewards.

        Args:
            n1 (int): The index of the first node. Must be less than n2.
            n2 (int): The index of the second node. Must be greater than n1.
            
        Returns:
            A tuple of two integers representing rewards:
            - If the move leads to FIXER winning, (1, -1)
            - If the move leads to neither player winning, (0, 0)
            - If the move is invalid, (-1, 0)
        """
        if self.adj_matrix[n1, n2] == 1:
            self.adj_matrix[n1, n2] = -1
            winner = self.get_winner()
            if winner == Player.FIXER:
                return 1, -1
            return 0, 0
        else: # invalid move case
            return -1, 0
    
    def get_observation(self) -> np.ndarray:
        """
        Returns the current observation of the game.

        Returns:
            A 1D numpy array representing the current observation of the game.
            The array contains the upper-triangle elements of the adjacency matrix, which are the only relevant parts of the graph.
        """
        return self.adj_matrix[np.triu_indices(3, k=1)]
        
    def take_action(self, move: np.ndarray) -> tuple[int, int]:
        """
        Takes an action in the game and returns the resulting rewards.

        Args:
            move (np.ndarray): A tuple of two integers representing the nodes to cut or fix.
            
        Returns:
            A tuple of two integers representing rewards:
            - If the move leads to CUTTER winning, (-1, 1)
            - If the move leads to FIXER winning, (1, -1)
            - If the move leads to neither player winning, (0, 0)
            - If the move is invalid, (0, -1) if CUTTER's turn or (-1, 0) if FIXER's turn.
        """
        assert len(move) == 2, "Move must be a tuple of length 2"

        min_node, max_node = min(move), max(move)

        if self.player_turn == Player.CUTTER:
            rewards = self.__cut(min_node, max_node)
        else:
            rewards = self.__fix(min_node, max_node)
        self.player_turn = Player.FIXER if self.player_turn == Player.CUTTER else Player.CUTTER
        return rewards
            


    

