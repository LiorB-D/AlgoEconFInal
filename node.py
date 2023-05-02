from __future__ import annotations

from enum import Enum
from typing import Set


class NodeType(Enum):
    NORMAL = 0
    S = 1
    T = 2


class Node:

    def __init__(self, node_type=NodeType.NORMAL):
        # A set of nodes that are adjacent to this node.
        self.adjacent_nodes : Set[Node] = set()
        self.node_type = node_type
        

    
    
    def absorb(self, node2 : Node):
        #remember to remove node2 from the graph after calling
        self.adjacent_nodes |= node2.adjacent_nodes
        self.adjacent_nodes.remove(self)
        self.adjacent_nodes.remove(node2)
        del node2

    @staticmethod
    def add_edge(node1 : Node, node2 : Node):
        node1.adjacent_nodes.add(node2)
        node2.adjacent_nodes.add(node1)
    
    @staticmethod
    def remove_edge(node1: Node, node2: Node):
        node1.adjacent_nodes.discard(node2)
        node2.adjacent_nodes.discard(node1)

    