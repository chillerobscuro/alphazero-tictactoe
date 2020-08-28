import numpy as np

class NN:
    def __init__(self, game):
        self.game = game

    def predict(self, board):
        pol = abs(np.random.randn(self.game.size**2))
        s = sum(pol)
        return [q/s for q in pol], np.random.rand()

    def train(self, train_examples):
        print('time to train the network!')
        pass