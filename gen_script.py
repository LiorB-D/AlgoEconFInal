import sys
import random
import networkx as nx
import numpy as np
from tqdm import tqdm

def generate_random_graph(num_nodes, edge_prob, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.fast_gnp_random_graph(num_nodes, edge_prob) # type: ignore
    while not nx.has_path(G, 0, 1):
        G = nx.fast_gnp_random_graph(num_nodes, edge_prob, seed=seed) # type: ignore
    return G

def generate_random_adjacency_matrices(num_graphs, seed=None):
    num_nodes = 20
    edge_probs = np.linspace(0.1, 0.9, num_graphs)
    adjacency_matrices = np.empty((num_graphs, num_nodes, num_nodes), dtype=np.int8)

    for i, p in enumerate(tqdm(edge_probs)):
        G = generate_random_graph(num_nodes, p, seed)
        adjacency_matrix = nx.to_numpy_array(G, dtype=int) # type: ignore
        adjacency_matrices[i] = adjacency_matrix
    return adjacency_matrices

SEED = int(sys.argv[2])
num_graphs = int(sys.argv[1])

adjacency_matrices = generate_random_adjacency_matrices(num_graphs, seed=SEED)

with open("test_data_" + str(num_graphs) + "_" + str(SEED) + ".npy", 'wb') as f:
    np.save(f, adjacency_matrices)

#print(adjacency_matrices.shape)

### Copy this code to load the data
# with open("test_data.npy", 'rb') as f:
#     adjacency_matrices = np.load(f)


