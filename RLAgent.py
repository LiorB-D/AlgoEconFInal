import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from shannon_switching_game import ShannonSwitchingGame

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

