from pvn import PolicyValueNet
from mcts import MonteCarloTreeSearch
from config import DEFAULT_MODEL_PATH


class Player:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, thinking_depth=1024):
        model = PolicyValueNet(model_path)
        self.mcts = MonteCarloTreeSearch(model)
        self.thinking_depth = thinking_depth

    def self_play(self):
        self.new_game()
        while not self.is_game_end():
            self.show()
            move = self.pick()
            self.set_move(*move)
        self.show()

    def vs(self, opponent):
        current = self
        current.new_game()
        opponent.new_game()
        while not current.is_game_end():
            current.show()
            move = current.pick()
            current.set_move(*move)
            opponent.set_move(*move)
            current, opponent = opponent, current
        current.show()
        if current.get_game_state() == 'draw':
            print('Draw game.')
        elif current != self:
            print('Player win.')
        else:
            print('Player lose.')

    def set_thinking_depth(self, depth):
        self.thinking_depth = depth

    def pick(self):
        x, y = self.mcts.pick(self.thinking_depth)
        return (x + 1, y + 1)

    def get_game_state(self):
        return self.mcts.get_game_state()

    def is_game_end(self):
        return self.mcts.is_game_end()

    def show(self):
        self.mcts.show()

    def new_game(self):
        self.mcts.new_game()

    def set_move(self, x, y):
        move = (x - 1, y - 1)
        self.mcts.set_move(move)

    def save(self, name):
        self.mcts.save_data('../model/{}.data'.format(name))
        self.mcts.save_model('../model/{}.model'.format(name))

    def load_data(self, path):
        self.mcts.load_data(path)

    def train(self, epochs=5, batch_size=128):
        self.mcts.train(epochs, batch_size)
