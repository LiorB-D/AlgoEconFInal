import pygame
import numpy as np
import random

# Define adjacency matrix
adj_matrix = np.array([[0, 1, 1, 0],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [0, 1, 1, 0]
                       ])





#function that takes in adjacency matrix indicies and removes the edge
def remove_edge(node1, node2):
    global edges
    global adj_matrix
    if adj_matrix[node1, node2] != -1:
        edges.remove((node1, node2))
        adj_matrix[node1, node2] = 0
        adj_matrix[node1, node2] = 0 
        return True
    return False
    
    

#function that takes in adjacency matrix indicies and reinforces the edge
def reinforce_edge(node1, node2):
    global adj_matrix
    if adj_matrix[node1, node2] != -1:
        adj_matrix[node1, node2] = -1
        return True
    return False


#function that draws all the nodes
def draw_nodes():
    global node_positions
    for position in node_positions:
        if position == (50, screen_height/2):
            pygame.draw.circle(screen, s_color, position, 15)
        elif position == (screen_width-50, screen_height/2):
            pygame.draw.circle(screen, s_color, position, 15)
        else:
            pygame.draw.circle(screen, node_color, position, 10)


#function that draws all the edges
def draw_edges():
    global edges
    for edge in edges:
        if adj_matrix[edge[0], edge[1]] == -1:
            pygame.draw.line(screen, reinforce_edge_color, node_positions[edge[0]], node_positions[edge[1]], 2)
        
        else:
            pygame.draw.line(screen, edge_color, node_positions[edge[0]], node_positions[edge[1]], 2)
    
       
       
       
       
screen_width, screen_height = 1000, 800
  
# Define your node positions
node_positions = []

#create S and T nodes at specific positions
spos = (50, screen_height/2)
tpos = (screen_width-50, screen_height/2)
node_positions.append(spos)
node_positions.append(tpos)

while len(node_positions) < len(adj_matrix):
        # Generate a random position for the new node
        x = random.randint(50, screen_width-50)
        y = random.randint(50, screen_height-50)
        pos = (x, y)

        # Check if the new node is at least min_distance away from all other nodes
        too_close = False
        for node_pos in node_positions:
            distance = np.sqrt((pos[0] - node_pos[0])**2 + (pos[1] - node_pos[1])**2)
            if distance < 20:
                too_close = True
                break
            if (np.abs(pos[0]- node_pos[0])) < 10 or (np.abs(pos[1]- node_pos[1])) < 10:
                too_close = True
                break

        # Add the new node to the list if it is not too close to any other nodes
        if not too_close:
            node_positions.append(pos)




    

# Define your node and edge colors
node_color = (255, 255, 255)
s_color = (0, 255, 0)
edge_color = (255, 255, 255)
reinforce_edge_color = (0, 0, 255)

# Initialize Pygame
pygame.init()

# Set the screen size and title
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Graph Visualization")



# Initialize and draw edges
edges = []
for i in range(adj_matrix.shape[0]):
    for j in range(i, adj_matrix.shape[1]):
        if adj_matrix[i, j] != 0:
            edge = (i, j)
            edges.append(edge)
            pygame.draw.line(screen, edge_color, node_positions[i], node_positions[j], 2)

# Draw nodes
draw_nodes()



# Update the display
pygame.display.flip()



# Wait for the user to close the window
player1_turn = True
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
                if distance < 3:
                    
                    if player1_turn:                                        
                        removed = remove_edge(edge[0], edge[1])
                        if removed:
                            player1_turn = False
                    else:
                        reinforced = reinforce_edge(edge[0], edge[1])
                        if reinforced:
                            player1_turn = True
                    
                    # Clear the screen and redraw everything except the removed edge
                    screen.fill((0, 0, 0))
                    
                    draw_edges()
                    
                    
                    for position in node_positions:
                        if position == (50, screen_height/2):
                            pygame.draw.circle(screen, s_color, position, 15)
                        elif position == (screen_width-50, screen_height/2):
                            pygame.draw.circle(screen, s_color, position, 15)
                        else:
                            pygame.draw.circle(screen, node_color, position, 10)
                            
                    
                    pygame.display.flip()
                    break 