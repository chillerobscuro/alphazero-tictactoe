from mcts import MCTS
import numpy as np
from random import shuffle


class Coach:
    """
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.num_iters = 2
        self.num_eps = 3
        self.temp_thresh = 5
        self.train_examples_history = []  # history of examples from numItersForTrainExamplesHistory latest iterations

    def execute_episode(self):
        training_examples = []
        board = self.game.get_init_board()
        self.current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            temp = int(episode_step < self.temp_thresh)  # todo why
            canon_board = self.game.get_canonical_board(board, self.current_player)
            print(f'\tep step {episode_step}, get action probs for board\n {self.game.get_canonical_board(board, 1)}')
            # calculate action probs
            pi = self.mcts.get_action_probs(canon_board, temp=temp)  # returns probability vector
            # todo get symmetries
            training_examples.append([canon_board, self.current_player, pi, None])
            chosen_action = np.random.choice(len(pi), p=pi)
            board, self.current_player = self.game.get_next_state(board, self.current_player, chosen_action)
            reward = self.game.check_game_ended(board, self.current_player)
            if reward != 0:  # return examples in form [board, policies, reward (flipped to match current_player)]
                print(f'\tep step {episode_step}, game ended! \n {self.game.get_canonical_board(board, 1)}')
                examples_w_rewards = [(x[0], x[2], reward * ((-1)**(x[1] != self.current_player))) for x in training_examples]
                return examples_w_rewards

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.num_iters + 1):
            # examples of the iteration
            iteration_train_examples = []

            for _ in range(self.num_eps):
                print(f'\non iteration {i} ep {_+1}, executing episode')
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                iteration_train_examples += self.execute_episode()

            # save the iteration examples to the history
            self.train_examples_history.append(iteration_train_examples)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)
            # pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)