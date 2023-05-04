import random

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from shannon_switching_game import Player, ShannonSwitchingGame


# Define the Experience class, which holds information about a state-action transition
class Experience:
    def __init__(self, adjM, action, r):
        self.state = np.copy(adjM)
        self.action = np.copy(action)
        self.reward = r
    
    def setSPrime(self, newAdj):
        self.sPrime = np.copy(newAdj)

# Define the QHandler class, which is responsible for the Q-Learning process
class QHandler:
    def __init__(self):
        self.expReplay = []  # Experience replay buffer
        self.setup_model()  # Set up the neural network model
        self.epsilon = 0  # Initial exploration rate
        self.discount = 0.15  # Discount factor for future rewards


    # Set up the neural network model for Q-Learning
    def setup_model(self):
        # Define the layers of the neural network
        layer_1 = layers.Dense(units = 200, input_dim = 190, activation = "relu")
        layer_2 = layers.Dense(units = 300, activation = "relu")
        layer_3 = layers.Dense(units = 300, activation = "relu")
        layer_4 = layers.Dense(units = 190, activation = "linear")

        # Create the model
        self.model = keras.Sequential([layer_1, layer_2, layer_3, layer_4])

        # Compile the model with mean squared error loss
        self.model.compile(
            loss='mean_squared_error'
        )

        # Update the target network
        self.updateTargetNetwork()
    
    # Clone the model to create a target network
    def updateTargetNetwork(self):
        self.targetModel = keras.models.clone_model(self.model)

    # Get the best move based on the current model
    def get_best_move(self, adjM):
        valid_moves = np.argwhere(np.triu(adjM, 1) == 1)
        valid_move_indices = self.get_valid_indices(valid_moves)  # Get the corresponding indices

        # Choose a random move with probability epsilon (exploration)
        if random.random() < self.epsilon:
            randInd = np.random.randint(0, len(valid_moves))
            return valid_moves[randInd], randInd 
        
        # Choose the move with the highest Q-value (exploitation)
        xs = np.asarray([self.adjMatrixToArray(adjM)])
        
        prediction = self.model.predict(xs)
        best_move_ind = np.argmax(prediction[0][valid_move_indices])
        best_move = valid_moves[best_move_ind]

        return best_move, best_move_ind

    # Train the Q-Network using the experience replay buffer
    def trainQNetwork(self):
        xs = []
        xPrimes = []

        # Process the experiences in the replay buffer
        for ind, exp in enumerate(self.expReplay):
            stateArr = self.adjMatrixToArray(exp.state)
            xs.append(stateArr)
            if not exp.reward == 0:
                xPrimes.append(stateArr)
            else:
                nextStateArr = self.adjMatrixToArray(exp.sPrime)
                xPrimes.append(nextStateArr)

        # Get the current and next state predictions
        ys = self.model.predict(np.array(xs), verbose = 0)
        ysTarget = self.targetModel.predict(np.array(xPrimes), verbose = 0)
        print(len(xs))
        print(len(ys))

        # Update the target values to train on
        for ind, exp in enumerate(self.expReplay):
            if not exp.reward == 0:
                print(exp.reward)
                ys[ind][self.matrixPositionToArrayIndex(exp.action[0],exp.action[1])] = exp.reward
            else:
                valid_moves = np.argwhere(exp.state == 1)
                valid_move_indices = self.get_valid_indices(valid_moves)
                nextStateArr = self.adjMatrixToArray(exp.sPrime)
                choices = ysTarget[ind][valid_move_indices]
                ys[ind][self.matrixPositionToArrayIndex(exp.action[0],exp.action[1])] = self.discount * np.max(choices)
                
               
        rng = np.random.default_rng()
        xsTensor = rng.choice(np.array(xs), 5000)
        ysTensor = rng.choice(np.array(ys), 5000)
        self.model.fit(xsTensor, ysTensor, epochs = 50, verbose = 1)
        
        

    # Convert the adjacency matrix to a flattened array, removing redundant entries
    def adjMatrixToArray(self, matrix):
        result = []
        # Add the lower triangular matrix of -100 to the input matrix, flatten the result, and keep non-negative elements
        result = (np.tril(-100 * np.ones((20,20))) + matrix).flatten()
        return result[result >= -2]

    # Convert a 1D array index to a 2D matrix position (row, col) in an upper triangular matrix
    def arrayIndexToMatrixPosition(self, index, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count == index:
                    return i, j  # Return the matrix position corresponding to the given index
                count += 1

    # Convert a 2D matrix position (row, col) to a 1D array index in an upper triangular matrix
    def matrixPositionToArrayIndex(self, row, col):
        # Calculate the index using the row and col values
        index = (row * (2 * 20 - row - 1)) // 2 + col - row - 1
        return index

    # Get the 1D array indices for the valid moves in a 2D matrix
    def get_valid_indices(self, valid_moves):
        result = []
        for m in valid_moves:
            # Convert each matrix position to its corresponding array index and append it to the result
            result.append(self.matrixPositionToArrayIndex(m[0], m[1]))
        return np.array(result)  # Return the valid indices as a NumPy array


class Experience:

    def __init__(self, state, action, reward, next_state, done) -> None:
        self.state = state # the state in which the action was taken
        self.action = action # the action taken
        self.reward = reward # this is the sum of the rewards from the action taken and the agents next action
        self.next_state = next_state # this is the state that the agent is in when it gets to make its next action
        self.done = done # this indicates whether the game ended after the action was taken 


class RLAgent:

    def __init__(self) -> None:
        self.initialize_agent_parameters()
        self.setup_q_network()
        self.initialize_experience_replay()


    def initialize_agent_parameters(self):
        # Initialize the agent's parameters, e.g. epsilon, learning rate, discount factor, etc.
        self.epsilon = 0.1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.1
        self.discount_factor = 0.1


    def setup_q_network(self):
        # Define the network architecture and initialize the network weights
        layer_1 = layers.Dense(units = 200, input_dim = 190, activation = "relu")
        layer_2 = layers.Dense(units = 300, activation = "relu")
        layer_3 = layers.Dense(units = 300, activation = "relu")
        layer_4 = layers.Dense(units = 190, activation = "linear")

        # Create the model
        self.model = keras.Sequential([layer_1, layer_2, layer_3, layer_4])

        # Compile the model with mean squared error loss
        self.model.compile(
            loss='mean_squared_error'
        )   

        self.updateTargetNetwork()


    def initialize_experience_replay(self):

        self.buffer_size = 10000
        self.batch_size = 16
        self.exp_replay = []


    def store_experience(self, state, action, reward, next_state, done):

        exp = Experience(state, action, reward, next_state, done)
        if len(self.exp_replay) >= self.buffer_size:
            self.exp_replay.pop(0)

        # Add the new experience to the buffer
        self.exp_replay.append(exp)


    def updateTargetNetwork(self):
        self.targetModel = keras.models.clone_model(self.model)


    def sample_experience(self):
        return random.sample(self.exp_replay, self.batch_size)
    

    # returns the index of the action to take
    def select_action(self, state):
        # Select an action using epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Select a random action
            return random.randint(0, 189)
        else:
            # Select the best action
            return np.argmax(self.model.predict(state))
        
    
    def train(self):
        if len(self.exp_replay) < self.batch_size:
            return

        experiences = self.sample_experience()
        
        states = np.array([exp.state for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        

        q_values_next = self.targetModel.predict(next_states)
        targets = self.model.predict(states)


        for i, exp in enumerate(experiences):
            target = exp.reward
            if not exp.done:
                target += self.discount_factor * np.max(q_values_next[i])
            targets[i][exp.action] = target

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def save_model(self, filepath):
        # Save the model weights to a file
        self.model.save_weights(filepath)
    

    def load_model(self, filepath):
        # Load the model weights from a file
        self.model.load_weights(filepath)


class MultiAgentEnvironment:

    def __init__(self):
        self.initialize_environment()
        self.initialize_agents()

    def sample_env(self, edge_prob = 0.5):
        G = nx.fast_gnp_random_graph(num_nodes, edge_prob) # type: ignore
        while not nx.has_path(G, 0, 1):
            G = nx.fast_gnp_random_graph(num_nodes, edge_prob, seed=seed) # type: ignore
        return nx.to_numpy_array(G, dtype=int) # type: ignore

    def initialize_environment(self):
        # Initialize your multi-agent environment here, e.g. custom environment, etc.
        rand = max(min(random.random(), 0.9), 0.1)
        sample = self.sample_env(edge_prob=rand)
        self.env : ShannonSwitchingGame = ShannonSwitchingGame(sample)
        

    def initialize_agents(self):
        # Initialize your agents, e.g. Agent 1, Agent 2, etc.
        self.fixer_agent = RLAgent() #FIXER
        self.cutter_agent = RLAgent() #CUTTER
        # Customize the agents' parameters or architectures as needed
    
    def agent_action_to_env_action(self, action_index, n=20):
        i, j = np.triu_indices(n, k=1)
        return int(i[action_index]), int(j[action_index])

    def run_episode(self):
        # Run an episode in the environment, interact with the environment, and train both agents
        self.initialize_environment()
        done = False
        prev_fixer_observation = None
        prev_fixer_action = None
        prev_fixer_rewards = 0

        prev_cutter_observation = None
        prev_cutter_action = None
        prev_cutter_rewards = 0
        

        while True:
            state = self.env.get_observation()
            if self.env.player_turn == Player.FIXER:
                if prev_fixer_action:
                    self.fixer_agent.store_experience(prev_fixer_observation, prev_fixer_action, prev_fixer_rewards, state, done)
                if done:
                    break

                action = self.fixer_agent.select_action(state)
                n1, n2 = self.agent_action_to_env_action(action)
                fixer_reward, cutter_reward, done = self.env.take_action((n1, n2))

                prev_fixer_rewards = fixer_reward
                prev_cutter_rewards += cutter_reward
                prev_fixer_observation = state
                prev_fixer_action = action
            else:
                if prev_cutter_action:
                    self.cutter_agent.store_experience(prev_cutter_observation, prev_cutter_action, prev_cutter_rewards, state, done)
                if done:
                    break

                action = self.cutter_agent.select_action(state)
                n1, n2 = self.agent_action_to_env_action(action)
                fixer_reward, cutter_reward, done = self.env.take_action((n1, n2))

                prev_cutter_rewards = cutter_reward
                prev_fixer_rewards += fixer_reward
                prev_cutter_observation = state
                prev_cutter_action = action

            
        self.fixer_agent.train()
        self.cutter_agent.train()

        # In each step of the episode:
            # 1. Get the current state for both agents
            # 2. Both agents select their actions
            # 3. Execute both agents' actions in the environment and obtain the next state, rewards, and done flags
            # 4. Store the experiences for each agent in their respective replay buffers
            # 5. Train both agents using their own experiences
        

    def train_agents(self, num_episodes):
        # Train both agents for a specified number of episodes
        for i in range(num_episodes):
            self.run_episode()
            print(f"Episode {i} finished")
        

    def evaluate_agents(self):
        # Evaluate the agents' performance in the environment
        pass

    def save_models(self, filepath1, filepath2):
        # Save the trained Q-network models for both agents
        self.fixer_agent.save_model(filepath1)
        self.cutter_agent.save_model(filepath2)

    def load_models(self, filepath1, filepath2):
        # Load pre-trained Q-network models for both agents
        self.fixer_agent.load_model(filepath1)
        self.cutter_agent.load_model(filepath2)


    