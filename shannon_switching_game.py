from collections import deque
from enum import Enum

import numpy as np


class Player(Enum): 
    FIXER = 1
    CUTTER = 2

### -1 = Reinforced Edge
### 0 = No Edge
### 1 = Normal Edge


class ShannonSwitchingGame:

    def __init__(self, adj_matrix : np.ndarray) -> None:
        """
        Initializes the ShannonSwitchingGame instance.

        Args:
        - adj_matrix: An adjacency matrix that describes the graph of the game.
        - source: The index of the source node.
        - target: The index of the target node.
        """
        n, m = adj_matrix.shape
        assert len(adj_matrix.shape) == 2, "Adjacency matrix must be 2D"
        assert n == m, "Adjacency matrix must be square"
        assert np.all(adj_matrix == adj_matrix.T), "Adjacency matrix must be symmetric"
        assert np.all(np.logical_or(adj_matrix == 0, adj_matrix == 1)), "Graph must only contain 0s or 1s to begin"
        assert n <= 20, "Adjacency matrix must be at most 20x20"
        
        self.num_nodes = n

        # make sure diagonal is all 0s
        np.fill_diagonal(adj_matrix, 0)

        pad = max(20 - n, 0)

        # Apply the padding
        self.adj_matrix : np.ndarray = np.pad(adj_matrix, ((0, pad), (0, pad)), 'constant', constant_values=0)

        self.source : int = 0
        self.target : int = 1
        self.player_turn : Player = Player.FIXER


    def valid_moves(self) -> np.ndarray:
        """
        Returns an array of valid moves for the current player.

        Returns:
        - An array of valid moves(array of position in adj matrix) for the current player.
        """
        return np.argwhere(np.triu(self.adj_matrix, 1) == 1)
    

    def get_winner(self):
        """
        Returns the winner of the game, if there is one.

        Returns:
        - The Player who won the game, or None if no one has won yet.
        """
        # first, bfs to check if reinforced path to target exists
        seen = [0] * self.num_nodes
        seen[self.source] = 1
        q = deque([self.source])
        while q:
            v = q.popleft()
            for w in range(self.num_nodes):
                if v != w and not seen[w] and self.adj_matrix[v, w] == -1:
                    if w == self.target:
                        return Player.FIXER
                    seen[w] = 1
                    q.append(w)
        
        # next, bfs to check if normal path to target exists
        seen = [0] * self.num_nodes
        seen[self.source] = 1
        q = deque([self.source])
        while q:
            v = q.popleft()
            for w in range(self.num_nodes):
                if v != w and not seen[w] and self.adj_matrix[v, w] in [1, -1]:
                    if w == self.target:
                        return None
                    seen[w] = 1
                    q.append(w)

        # otherwise, s and t are in different components and cutter wins
        return Player.CUTTER
    

    def __cut(self, n1: int, n2: int) -> bool:
        """
        Cuts the edge between nodes n1 and n2 and returns rewards.

        Args:
            n1 (int): The index of the first node. Must be less than n2.
            n2 (int): The index of the second node. Must be greater than n1.
            
        Returns:
            True if the move is valid, False otherwise.
        """
        if self.adj_matrix[n1, n2] == 1:
            self.adj_matrix[n1, n2] = 0
            self.adj_matrix[n2, n1] = 0
            return True
        else:
            print("Invalid Move Data: ")
            print(n1)
            print(n2)
            print(self.adj_matrix[n1,n2])
            print(self.adj_matrix)
        return False
        
    def __fix(self, n1: int, n2: int) -> bool:
        """
        Fixes the edge between nodes n1 and n2 and returns rewards.

        Args:
            n1 (int): The index of the first node. Must be less than n2.
            n2 (int): The index of the second node. Must be greater than n1.
            
        Returns:
            True if the move is valid, False otherwise.
        """
        if self.adj_matrix[n1, n2] == 1:
            self.adj_matrix[n1, n2] = -1
            self.adj_matrix[n2, n1] = -1
            return True
        return False
    
    def get_observation(self) -> np.ndarray:
        """
        Returns the current observation of the game.

        Returns:
            A 1D numpy array representing the current observation of the game.
            The array contains the upper-triangle elements of the adjacency matrix, which are the only relevant parts of the graph.
        """ 
        return self.adj_matrix[np.triu_indices(3, k=1)]
        
    def take_action(self, move: np.ndarray):
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
            if not self.__cut(min_node, max_node):
                return 0, -1
        else:
            if not self.__fix(min_node, max_node):
                return -1, 0

        winner = self.get_winner()
        if winner == Player.CUTTER:
            return -1, 1
        if winner == Player.FIXER:
            return 1, -1
        
        self.player_turn = Player.FIXER if self.player_turn == Player.CUTTER else Player.CUTTER
        return 0, 0
