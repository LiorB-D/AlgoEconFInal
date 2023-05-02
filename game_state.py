import random
from enum import Enum
from typing import List, Set

from node import Node, NodeType


class Player(Enum): 
    FIXER = 1
    CUTTER = 2


class GameState:

    def __init__(self, graph : Set[Node] | List[Node], s : Node, t : Node):
        self.active_nodes : Set[Node] = set(graph)

        s.node_type = NodeType.S
        t.node_type = NodeType.T

        self.s : Node = s
        self.t : Node = t

        self.player_turn : Player = random.choice([Player.FIXER, Player.CUTTER])

    
    def winner(self) -> Player | None:
        if (self.s in self.active_nodes and not self.t in self.active_nodes) or (self.t in self.active_nodes and not self.s in self.active_nodes):
            return Player.FIXER
        elif True: #TODO @ryan check if cutter wins
            return Player.CUTTER
        else:
            #returns none if no one has won yet
            return None
        
    
    def __cut(self, node1 : Node, node2: Node):
        Node.remove_edge(node1, node2)
        # TODO: could optmize to prune


    def __fix(self, node1 : Node, node2: Node):
        node1.absorb(node2)
        self.active_nodes.discard(node2)
        pass

    def valid_mode(self, node1 : Node, node2: Node):
        if node1 in node2.adjacent_nodes and node2 in node1.adjacent_nodes:
            return True
        return False
    
    
    def move(self, node_1, node_2) -> int:
        if not self.valid_mode(node_1, node_2):
            raise Exception("Invalid move")
        
        if self.player_turn == Player.CUTTER:
            self.__cut(node_1, node_2)
        else:
            self.__fix(node_1, node_2)
        
        win = self.winner()
        reward = 0
        if win == Player.FIXER and self.player_turn == Player.FIXER:
            reward = 1
        elif win == Player.CUTTER and self.player_turn == Player.CUTTER:
            reward = 1
        else:
            reward = -1
        
        self.player_turn = Player.FIXER if self.player_turn == Player.CUTTER else Player.CUTTER
            
        return reward

    

    



    


    

