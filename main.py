from game import Game
from nn import NNWrapper
from coach import Coach
from utils import printstr


def main():
    g = Game()
    if load_model:
        nn = ''
    else:
        nn = NNWrapper(g)
    args = []
    c = Coach(g, nn, args)
    c.learn()
    pass


if __name__ == '__main__':
    print(printstr)
    load_model = 0
    main()

'''
todo - 
make nn
make human player
add arguments from command line
'''