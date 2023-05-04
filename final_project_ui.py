import pygame
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from shannon_switching_game import ShannonSwitchingGame, Player
from RLAgent import QHandler, Experience


class play_game:
    
    def __init__(self, adj_matrix : np.ndarray) -> None:
        """
        Initializes the play_game instance.
        Args:
        - adj_matrix: An adjacency matrix that describes the graph of the game.
        """

        self.SSG = ShannonSwitchingGame(adj_matrix)
        self.node_positions = []
        self.edges = []
        self.screen_width = 1000
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.node_color = (255, 255, 255)
        self.s_color = (0, 255, 0)
        self.edge_color = (255, 255, 255)
        self.reinforce_edge_color = (0, 0, 255)
            
       
        #add all the nodes to node_positions
        spos = (50, self.screen_height/2)
        tpos = (self.screen_width-50, self.screen_height/2)
        self.node_positions.append(spos)
        self.node_positions.append(tpos)

        #num_nodes = 0

        # for i in range(len(self.SSG.adj_matrix[0])):
        #     for j in range(len(self.SSG.adj_matrix[0])):
        #         if np.sum(self.SSG.adj_matrix[i]) > 0 or (self.SSG.adj_matrix[j][i]) > 0:
        #             num_nodes +=1
        #             break
                    
        while len(self.node_positions) < (self.SSG.num_nodes):
                # Generate a random position for the new node
                x = random.randint(50, self.screen_width-50)
                y = random.randint(50, self.screen_height-50)
                pos = (x, y)

                # Check if the new node is at least min_distance away from all other nodes
                too_close = False
                for node_pos in self.node_positions:
                    distance = np.sqrt((pos[0] - node_pos[0])**2 + (pos[1] - node_pos[1])**2)
                    if distance < 50:
                        too_close = True
                        break
                    if (np.abs(pos[0]- node_pos[0])) < 10 or (np.abs(pos[1]- node_pos[1])) < 10:
                        too_close = True
                        break

                # Add the new node to the list if it is not too close to any other nodes
                if not too_close:
                    self.node_positions.append(pos)
                    
        #add all the edges to self.edges
        for i in range(self.SSG.adj_matrix.shape[0]):
            for j in range(i, self.SSG.adj_matrix.shape[1]):
                if self.SSG.adj_matrix[i, j] != 0:
                    
                    edge = (i, j)
                    self.edges.append(edge)
                    pygame.draw.line(self.screen, self.edge_color, self.node_positions[i], self.node_positions[j], 2)


        self.QRL_cutter = QHandler()
        self.QRL_cutter.model = tf.keras.models.load_model("Cutter_Model")

        self.human_cutter = False
        self.human_turn = True


    def draw_nodes(self) -> None:
        
        for position in self.node_positions:
            if position == (50, self.screen_height/2):
                pygame.draw.circle(self.screen, self.s_color, position, 15)
            elif position == (self.screen_width-50, self.screen_height/2):
                pygame.draw.circle(self.screen, self.s_color, position, 15)
            else:
                pygame.draw.circle(self.screen, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), position, 10)
        font = pygame.font.Font(None, 36)

        # create a surface object with the text to be displayed
        s_text = font.render("S", True, (0, 0, 0))
        t_text = font.render("T", True, (0, 0, 0))

        # get the dimensions of the text surface
        s_text_rect = s_text.get_rect()
        t_text_rect = t_text.get_rect()
        
        # set the position of the text on the screen
        s_text_rect.center = (50, self.screen_height/2)
        t_text_rect.center = (self.screen_width-50, self.screen_height/2)

        # blit the text surface to the screen
        self.screen.blit(s_text, s_text_rect)
        self.screen.blit(t_text, t_text_rect)
                
                       

    def draw_edges(self):
        for edge in self.edges:
            if self.SSG.adj_matrix[edge[0], edge[1]] == -1:
                pygame.draw.line(self.screen, self.reinforce_edge_color, self.node_positions[edge[0]], self.node_positions[edge[1]], 2)
            
            else:
                pygame.draw.line(self.screen, self.edge_color, self.node_positions[edge[0]], self.node_positions[edge[1]], 2)
                
    
    def draw_background(self):
        background_image = pygame.image.load("starImage.jpg").convert()
        self.screen.blit(background_image, (0, 0))
        
       
       
    def run_game(self):
        # Initialize Pygame
        pygame.init()

        # Set the screen size and title
        pygame.display.set_caption("Graph Visualization")
        
        self.draw_background()
        
        self.draw_edges()
        # Draw nodes
        self.draw_nodes()
        

        

        # Update the display
        pygame.display.flip()

        running = True
        found_winner = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and found_winner == False:
                    mouse_pos = pygame.mouse.get_pos()
                    for edge in self.edges:
                        node1_pos = self.node_positions[edge[0]]
                        node2_pos = self.node_positions[edge[1]]
                        # Calculate the distance between the mouse position and the edge
                        distance = abs((node2_pos[1] - node1_pos[1]) * mouse_pos[0] -
                                    (node2_pos[0] - node1_pos[0]) * mouse_pos[1] +
                                    node2_pos[0] * node1_pos[1] - node2_pos[1] * node1_pos[0]) / \
                                np.sqrt((node2_pos[1] - node1_pos[1]) ** 2 +
                                        (node2_pos[0] - node1_pos[0]) ** 2)
                                
                                
                        # If the distance is less than a certain threshold, accept the click
                        if distance < 5:
                            move = (edge[0], edge[1])
                            reward = self.SSG.take_action(move)                         
                            
                            if reward == (-1, 1):
                                found_winner = True
                                self.edges.remove(edge)
                                print("Cutter Wins")
                            
                            if self.SSG.player_turn == Player.FIXER and reward == (0,0):
                                self.edges.remove(edge)
                            
                            if reward == (1, -1):
                                found_winner = True
                                print("Fixer Wins")
                                
                            # Clear the screen and redraw everything except the removed edge
                            self.screen.fill((0, 0, 0))
                            
                            self.draw_background()
                            self.draw_edges()
                            self.draw_nodes()

                            pygame.display.flip()

                            time.sleep(1)


                            # now ai makes a move
                            if self.SSG.player_turn == Player.CUTTER and found_winner == False:
                                ai_move = self.QRL_cutter.get_best_move(self.SSG.adj_matrix)
                                reward = self.SSG.take_action(ai_move)
                                
                                edge = (ai_move[0], ai_move[1])

                                if reward == (-1, 1):
                                    found_winner = True
                                    self.edges.remove(edge)
                                    print("Cutter Wins")
                                
                                elif reward == (0,0):
                                    self.edges.remove(edge)
                                
                                elif reward == (1, -1):
                                    found_winner = True
                                    print("Fixer Wins")
                                
                                self.draw_background()
                                self.draw_edges()
                                self.draw_nodes()

                                pygame.display.flip()


                            
                            break 



if __name__ == '__main__':
    adj_matrix = np.array([[0, 1, 1, 0],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [0, 1, 1, 0]
                       ])
    
    matrix = np.zeros((10, 10))

    # fill the upper triangular part of the matrix with random 0's and 1's
    for i in range(10):
        for j in range(i, 10):
            matrix[i, j] = np.random.randint(2)

    # set the lower triangular part of the matrix to be equal to the upper triangular part
    for i in range(1, 10):
        for j in range(i):
            matrix[i, j] = matrix[j, i]
            
    game = play_game(matrix)
    game.run_game()
