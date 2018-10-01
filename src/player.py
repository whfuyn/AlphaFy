import time
from pvn import PolicyValueNet
from mcts import MonteCarloTreeSearch


class Player:
    def __init__(self, model=None, thinking_depth=1024, config=dict()):
        if model is None:
            pvn = PolicyValueNet(None, config)
            self.mcts = MonteCarloTreeSearch(pvn)
        elif isinstance(model, str):
            self.load_model(model)
        else:
            self.mcts = MonteCarloTreeSearch(model)
        self.thinking_depth = thinking_depth

    def self_play(self, show_board=True):
        self.new_game()
        while not self.is_game_end():
            if show_board:
                self.show()
            x, y = self.pick()
            self.set_move(x, y)
        if show_board:
            self.show()

    def vs_user(self, first_hand=False):
        self.new_game()
        while not self.is_game_end():
            self.show()
            if first_hand:
                self.get_user_move()
            else:
                x, y = self.pick()
                self.set_move(x, y)
            first_hand = not first_hand
        self.show()
        if self.get_game_state() == 'draw':
            print('Draw game.')
        elif first_hand:
            print('AI win.')
        else:
            print('User win.')

    def get_user_move(self):
        while True:
            try:
                cmd = input("move: ")
                x, y = eval(cmd)
                self.set_move(x, y)
            except KeyboardInterrupt as e:
                raise e
            except:
                print("Bad input, try again.")

    def vs(self, opponent, show_board=True):
        current = self
        current.new_game()
        opponent.new_game()
        while not current.is_game_end():
            if show_board:
                current.show()
            x, y = current.pick()
            current.set_move(x, y)
            opponent.set_move(x, y)
            current, opponent = opponent, current
        if show_board:
            current.show()
        if current.get_game_state() == 'draw':
            return 'Draw'
        elif current != self:
            return 'Win'
        else:
            return 'Lose'

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

    def save(self, name, override=False):
        self.save_data(name, override)
        self.save_model(name, override)

    def save_data(self, name, override=False):
        self.mcts.save_data('../data/{}.data'.format(name), override)

    def save_model(self, name, override=False):
        self.mcts.save_model('../model/{}.model'.format(name), override)

    def load_data(self, name='latest', merge=True):
        self.mcts.load_data('../data/{}.data'.format(name), merge)

    def load_model(self, name='latest'):
        model = PolicyValueNet('../model/{}.model'.format(name))
        self.mcts = MonteCarloTreeSearch(model)

    def train(self, epochs=5, batch_size=128):
        if self.mcts.train_data[0].shape[0] == 0:
            raise RuntimeError('player: Data is unloaded.')
        self.mcts.train(epochs, batch_size)

    def clear_train_data(self):
        self.mcts.clear_train_data()

    def evolve(self, epochs=10):
        for i in range(epochs):
            if i > 0:
                self.clear_train_data()
            timestamp = time.strftime('%Y%m%d%H%M%S')
            self.save_model(timestamp)
            opponent = Player(timestamp)
            opponent.disable_guide()
            n = 16
            while True:
                print('Self playing: ', end='')
                for i in range(n):
                    # self.enable_guide()
                    opponent.set_thinking_depth(32)
                    self.set_thinking_depth(32)
                    self.self_play(show_board=False)
                    print(i + 1, end=' ')
                    if (i + 1) % 64 == 0:
                        print('')
                        timestamp = time.strftime('%Y%m%d%H%M%S')
                        self.save(timestamp, override=True)
                        print('')
                print('done.')
                self.train(5, 128)
                score = dict(Win=0, Lose=0, Draw=0)
                opponent.set_thinking_depth(64)
                self.set_thinking_depth(64)
                self.disable_guide()
                for i in range(10):
                    score[self.vs(opponent, show_board=True)] += 1
                for i in range(10):
                    result = opponent.vs(self, show_board=True)
                    result = \
                        'Win' if result == 'Lose' else \
                        'Lose' if result == 'Win' else \
                        'Draw'
                    score[result] += 1
                print(score)
                if score['Lose'] + 3 <= score['Win']:
                    print('Evlove succeed.')
                    break
                else:
                    self.load_model(timestamp)
                    self.load_data(timestamp)
                n *= 2

    def benchmark(self, opponent):
            self.disable_guide()
            opponent.disable_guide()
            score = dict(Win=0, Lose=0, Draw=0)
            for i in range(10):
                score[self.vs(opponent, show_board=True)] += 1
            for i in range(10):
                result = opponent.vs(self, show_board=True)
                result = \
                    'Win' if result == 'Lose' else \
                    'Lose' if result == 'Win' else \
                    'Draw'
                score[result] += 1
            print(score)

    def enable_guide(self):
        self.mcts.enable_guide()

    def disable_guide(self):
        self.mcts.disable_guide()
