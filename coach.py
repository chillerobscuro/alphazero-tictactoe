from mcts import MCTS
import numpy as np


class Coach():
    """
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def execute_episode(self):
        training_examples = []
        board = self.game.get_init_board()
        self.current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            temp = int(episode_step < self.args.tempThreshold)
            canon_board = self.game.get_canonical_board(board, self.current_player)
            # calculate action probs
            pi = self.mcts.get_action_probs(canon_board, temp=temp)  # returns todo which 2 things?
            # todo get symmetries
            training_examples.append([canon_board, self.current_player, pi, None])
            chosen_action = np.random.choice(len(pi), p=pi)
            board, self.current_player = self.game.get_next_state(board, self.current_player, chosen_action)
            reward = self.game.check_game_ended(board, self.current_player)
            if reward != 0:
                # return examples in form [board, policies, reward (flipped to match current_player)]
                examples_w_rewards = [(x[0], x[2], reward * ((-1)**(x[1] != self.current_player))) for x in training_examples]
                return examples_w_rewards
