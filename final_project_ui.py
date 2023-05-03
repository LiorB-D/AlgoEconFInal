import pygame
import numpy as np
import random

# Define your adjacency matrix
adj_matrix = np.array([[0, 1, 1, 0,0],
                       [1, 0, 1, 1,1],
                       [1, 1, 0, 1,1],
                       [0, 1, 1, 0,1],
                       [1,0,0,0,1]])

# Define your node positions
node_positions = []
for each in adj_matrix:
    node_positions.append((random.randint(10, 790), random.randint(10, 590)))

# Define your node and edge colors
node_color = (255, 255, 255)
edge_color = (255, 255, 255)

# Initialize Pygame
pygame.init()

# Set the screen size and title
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Graph Visualization")

# Draw your nodes
for position in node_positions:
    pygame.draw.circle(screen, node_color, position, 10)

# Draw your edges
edges = []
for i in range(adj_matrix.shape[0]):
    for j in range(i, adj_matrix.shape[1]):
        if adj_matrix[i, j] == 1:
            edge = (i, j)
            edges.append(edge)
            pygame.draw.line(screen, edge_color, node_positions[i], node_positions[j], 2)

# Update the display
pygame.display.flip()

# Wait for the user to close the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for edge in edges:
                node1_pos = node_positions[edge[0]]
                node2_pos = node_positions[edge[1]]
                # Calculate the distance between the mouse position and the edge
                distance = abs((node2_pos[1] - node1_pos[1]) * mouse_pos[0] -
                               (node2_pos[0] - node1_pos[0]) * mouse_pos[1] +
                               node2_pos[0] * node1_pos[1] - node2_pos[1] * node1_pos[0]) / \
                           np.sqrt((node2_pos[1] - node1_pos[1]) ** 2 +
                                   (node2_pos[0] - node1_pos[0]) ** 2)
                # If the distance is less than a certain threshold, remove the edge
                if distance < 5:
                    adj_matrix[edge[0], edge[1]] = 0
                    adj_matrix[edge[1], edge[0]] = 0
                    edges.remove(edge)
                    # Clear the screen and redraw everything except the removed edge
                    screen.fill((0, 0, 0))
                    for position in node_positions:
                        pygame.draw.circle(screen, node_color, position, 10)
                    for e in edges:
                        pygame.draw.line(screen, edge_color, node_positions[e[0]], node_positions[e[1]], 2)
                    pygame.display.flip()
                    break  # Exit the loop since we already found and removed the edge