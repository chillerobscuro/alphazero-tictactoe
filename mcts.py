import numpy as np
import math


class MCTS():
    """
    Monte Carlo Tree Search Class
    """

    def __init__(self, game, nnet, args, num_mcts_sims=10, cpct=.5, temp=.5):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.cpct = cpct
        self.num_mcts_sims = num_mcts_sims
        self.temp = temp
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.getGameEnded ended for board s
        self.Vs = {}   # stores game.getValidMoves for board s

    def get_action_probs(self, canonical_board):
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.num_mcts_sims):
            self.search(canonical_board)

        s = self.game.string_rep(canonical_board)
        # for each possible action, return the count we've taken it from this state
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if self.temp == 0:  # Just pick the action we've taken from the state most often
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_actions)  # if we've been to 2 states the most, pick one randomly
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs  # return array where prob of best move is 100, rest 0

        counts = [x ** (1. / self.temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs



    def search(self, canonical_board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.string_rep(canonical_board)
        if s not in self.Es:  # haven't checked if game ended here yet
            self.Es[s] = self.game.check_game_ended(canonical_board, 1)  # player always 1 because canonical board
        if self.Es[s] != 0:  # game ended, so we have reached a terminal node, return and break recursive self.search()
            return -self.Es[s]  # return game value

        if s not in self.Ps:  # we haven't calculated policy for this state yet
            self.Ps[s], v = self.nnet.predict(canonical_board)  # return policy vector, value of current board
            valid_moves = self.game.get_valid_moves(canonical_board, 1)  # return binary array of which moves allowed
            self.Ps[s] = self.Ps[s] * valid_moves  # mask invalid moves
            sum_this_state_policy = np.sum(self.Ps[s])
            if sum_this_state_policy > 0:
                self.Ps[s] /= sum_this_state_policy  # renormalize policy vector
            else:
                print('oh no there are no valid moves you dumb bish')

            self.Vs[s] = valid_moves  # store valid moves from this board state
            self.Ns[s] = 0  # initialize count of how many times we've been in this state
            return -v  # value of board from other player's perspective # but why?


        valid_moves = self.Vs[s]
        current_best = -100
        best_action = -1
        # compute Upper Confidence Bound and pick action with highest value
        for a in range(self.game.get_action_size()):
            if valid_moves[a]:  # if this move is allowed
                if (s, a) in self.Qsa:  # if we've already calculated Q values for this board state and action,
                                        # then we just need to update the Q value in line
                    # from paper: U(s,a) = Q(s,a)+cpuct⋅P(s,a)⋅√ΣbN(s,b)/1+N(s,a)
                    u = self.Qsa[(s, a)] + self.cpct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:  # We haven't computed the Q values yet, so compute with policy vector and Ns[s]
                    u = self.cpct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > current_best:  # Best action we've seen so far for this board
                    current_best = u
                    best_action = a

        a = best_action  # the action we've chosen is the higest from our loop over all possible actions from s
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_canonical_board = self.game.get_canonical_board(next_s, next_player)

        v = self.search(next_canonical_board)  # this will be called recursively until self.search returns game value

        if (s, a) in self.Qsa:  # We have q values for this board and move so just update
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[s, a] + v) / self.Nsa[(s, a) + 1]
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v  # Note that we return the negative value of the state. This is because alternate levels in the search
                   # tree are from the perspective of different players. Since v∈[−1,1], −v is the value of the current
                   # board from the perspective of the other player.
