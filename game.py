import numpy as np


class Game:
    """
    TicTacToe game class
    """

    def __init__(self, size=5):
        self.size = size
        self.winning_inds = self.get_winning_inds()

    def get_init_board(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return np.array([[0]*self.size for x in range(self.size)])

    def get_board_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.size, self.size

    def get_action_size(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.size**2

    def get_next_state(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        flat = board.flatten()
        flat[action] = player
        board = flat.reshape(self.size, self.size)
        return board, -player

    def get_valid_moves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board,
                        0 for invalid moves
        """
        flat = board.flatten()
        return flat == 0

    def check_game_ended(self, board, player):
        """
        Input:
            board: current board in canonical form
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """  # todo make sure this is working right
        winner = 0
        for x in self.winning_inds:
            vals = board.flatten()[x]
            if all(vals == 1):
                winner = 1
            if all(vals == -1):
                winner = -1
            if sum(self.get_valid_moves(board)) == 0:
                winner = 1e-8
        return winner

    def get_winning_inds(self):
        sz = self.size
        winning_inds = []
        for i in range(sz):
            winning_inds.append(range(i*sz, i*sz+sz))      # add all rows
            winning_inds.append(range(i, sz**2)[::sz])     # add all columns
        winning_inds.append(range(sz**2)[::sz+1])          # add diagonal from top left
        winning_inds.append(range(sz-1, sz**2-1)[::sz-1])  # add diagonal from top right
        return winning_inds

    def get_canonical_board(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board * player

    def get_symmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        assert (len(pi) == self.size ** 2)
        pi_board = np.reshape(pi, (self.size, self.size))
        symmetrical_training_examples = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i)      # flip the board 90 degrees 8 times
                new_pi = np.rot90(pi_board, i)  # do the same for p vector
                if j:                           # half of the time we'll mirror the board (and vector)
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetrical_training_examples += [(new_b, list(new_pi.ravel()))]
        return symmetrical_training_examples    # returns 8 symmetrical iterations for every game state

    def string_rep(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board.flatten())[1:-1]
