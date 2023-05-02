import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random



class Experience:
    def __init__(self, action, adjM):
        self.state = adjM
        self.action = action
        self.reward = 0
        self.sPrime = []
    
    def setSPrime(self, newAdj):
        self.sPrime = newAdj

class QHandler:
    def __init__(self, gameState):
        self.expReplay = []
        self.setup_Model()
        self.epsilon = 1
        self.discount = 0.15
        self.game_state = gameState
    

    def setup_model(self):
        layer_1 = layers.Dense(units = 200, input_dim = 190, activation = "relu")
        layer_2 = layers.Dense(units = 300, activation = "relu")
        layer_3 = layers.Dense(units = 300, activation = "relu")
        layer_4 = layers.Dense(units = 190, activation = "linear")
        self.model = keras.Sequential([layer_1, layer_2, layer_3, layer_4])

        self.model.compile(
            loss='mean_squared_error'
        )

    def get_best_move(self, adjM):
        valid_moves = self.game_state.valid_moves()
        valid_move_indices = self.get_valid_indices(valid_moves)

        if random.random() < self.epsilon:
            return np.random.choice(valid_moves, 1)
        
        xs = [self.adjMatrixToArray(adjM)]
        prediction = self.model.predict(xs)

        best_move_ind = np.argmax(prediction[valid_move_indices])
        best_move = valid_moves[best_move_ind]

        return best_move, best_move_ind

    def trainQNetwork(self):
        xs = []
        ys = []
        for ind, exp in enumerate(self.expReplay):
            if (1 + ind) / len(self.expReplay) < random.random(): # More recent experiences are more likely to be chosen
                stateArr = self.matrixPositionToArrayIndex(exp.s)
                xs.append(stateArr)
                y = self.model.predict(np.array([stateArr]))[0]
                if not exp.r == 1:
                    y[self.matrixPositionToArrayIndex(exp.a[0],exp.a[1])] = exp.r
                else:
                    nextStateArr = self.matrixPositionToArrayIndex(exp.sPrime)
                    y[self.matrixPositionToArrayIndex(exp.a[0],exp.a[1])] = self.discount * np.max(self.model.predict(np.array([nextStateArr])))
                
                ys.append(y)


        rng = np.random.default_rng()
        xsTensor = rng.choice(np.array(xs), 5000)
        ysTensor = rng.choice(np.array(ys), 5000)
        self.model.fit(xsTensor, ysTensor, epochs = 500, verbose = 1)
        
        

    def adjMatrixToArray(self, matrix):
        result = []

        for i in range(20):
            for j in range(i+1, 20):
                result.append(matrix[i, j])

        return np.array(result)

    def arrayIndexToMatrixPosition(self, index, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count == index:
                    return i, j
                count += 1

    def matrixPositionToArrayIndex(row, col):

        index = (row * (2 * 20 - row - 1)) // 2 + col - row - 1
        return index

    def get_valid_indices(self, valid_move):
        result = []
        for m in valid_move:
            result.append(self.matrixPositionToArrayIndex(m[0], m[1]))
        return np.array(result)
