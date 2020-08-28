from game import Game
from nn import NN
from coach import Coach

from_dir = 0

def main():
    g = Game()
    if load_model:
        nn = from_dir
    else:
        nn = NN(g)
    args = []
    c = Coach(g, nn, args)
    c.learn()
    pass


if __name__ == '__main__':
    load_model = 0
    main()