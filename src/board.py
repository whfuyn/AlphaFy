import itertools
import numpy as np
from config import BOARD_SHAPE, NUM_CONNECTED_TO_WIN


class Board:
    def __init__(self, board=None, guide_enabled=False):
        if board is None:
            self.board = np.zeros(BOARD_SHAPE, dtype=np.int32)
            self.game_state = 'playing'
            self.available_moves = np.where(self.board == 0)
            self.last_move = None
            self.total_move = 0
            self.current_player = 1
        else:
            self.board = board
        self.guide_enabled = guide_enabled

    def get_available_moves(self):
        if self.game_state != 'playing':
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        if self.guide_enabled:
            threat_moves = self.get_all_threat_moves()
            if len(threat_moves) != 0:
                xs = np.array([p[0] for p in threat_moves])
                ys = np.array([p[1] for p in threat_moves])
                return (xs, ys)
        # It return (xs, ys) instead of xys, surprisingly.
        return np.where(self.board == 0)

    # Note that this function won't change self.board.
    # It return a new board instead.
    def put(self, move):
        if self.game_state != 'playing':
            raise RuntimeError(
                'The Game has ended with state: {}'.format(self.game_state)
            )
        if self.board[move] != 0:
            raise ValueError('Positon {} has been occupied.'.format(move))
        data = np.negative(self.board)
        data[move] = -1
        new_board = Board(data, self.guide_enabled)
        new_board.game_state = new_board.check_move(move)
        new_board.available_moves = new_board.get_available_moves()
        new_board.last_move = move
        new_board.total_move = self.total_move + 1
        new_board.current_player = (-1) * self.current_player
        return new_board

    def get_board_data(self):
        board_data = np.dstack([self.board == 1, self.board == -1])
        return board_data.astype(np.float32)

    def check_move(self, move):
        # Get pep8 warning when using lambda expr
        def is_vaild(x, y):
            X, Y = BOARD_SHAPE
            return x >= 0 and x < X and y >= 0 and y < Y

        color = self.board[move]
        steps = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in steps:
            x, y = move
            x += dx
            y += dy
            cnt = 1
            while is_vaild(x, y) and self.board[x, y] == color:
                cnt += 1
                x += dx
                y += dy
            x, y = move
            x -= dx
            y -= dy
            while is_vaild(x, y) and self.board[x, y] == color:
                cnt += 1
                x -= dx
                y -= dy
            if cnt >= NUM_CONNECTED_TO_WIN:
                return 'lose'

        if len(list(zip(*np.where(self.board == 0)))) == 0:
            return 'draw'
        return 'playing'

    # This UI is from github project with modifications.
    # Too lazy to design an UI for myself..
    def __str__(self):
        out = ""
        size = BOARD_SHAPE[0]

        s = '123456789ABCDEF'
        letters = list(s)[:size]
        numbers = list(reversed(s))[:size]

        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"

        out += (label_letters + label_boundry)
        shape = {0: '.', 1: 'O', -1: 'X'}
        for i in range(size - 1, -1, -1):
            line = ""
            line += (str("%2d" % (i + 1)) + " |" + " ")
            for j in range(size):
                line += shape[self.board[i, j] * self.current_player]
                if (i, j) == self.last_move:
                    line += ')'
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out

    def get_all_threat_moves(self):
        offensive, deffensive = [], []
        fatal, threat, weak = [], [], []
        opponent_threat = []
        for p in zip(*np.where(self.board == 1)):
            f, t, w = self.get_threat_moves(p, 1)
            fatal.extend(f)
            threat.extend(t)
            weak.extend(w)
        for p in zip(*np.where(self.board == -1)):
            f, t, _ = self.get_threat_moves(p, -1)
            opponent_threat.extend(f or t)

        if len(fatal) != 0:
            offensive.extend(fatal)
        elif len(threat) != 0:
            offensive.extend(threat)
        elif len(opponent_threat) != 0:
            deffensive.extend(opponent_threat)
            deffensive.extend(weak)
        moves = offensive or deffensive

        return list(set(moves))

    def get_threat_moves(self, xy, c):
        fatal = []
        threat = []
        weak = []

        steps = [
            (1, 0), (0, 1), (1, 1), (1, -1),
            (-1, 0), (0, -1), (-1, -1), (-1, 1)
        ]
        x, y = xy
        for dx, dy in steps:
            # Defined in here for capturing dx, dy.
            def is_vaild(x, y):
                X, Y = BOARD_SHAPE
                return x >= 0 and x < X and y >= 0 and y < Y

            def is_stone(k):
                return is_vaild(x + k * dx, y + k * dy) \
                    and self.board[x + k * dx, y + k * dy] == c

            def is_empty(k):
                return is_vaild(x + k * dx, y + k * dy) \
                    and self.board[x + k * dx, y + k * dy] == 0

            def is_block(k):
                return (not is_vaild(x + k * dx, y + k * dy)) \
                    or self.board[x + k * dx, y + k * dy] == -c

            # four XOOOO_
            if sum(map(is_stone, range(5))) == 4:
                k = next(filter(is_empty, range(5)), None)
                if k is not None:
                    threat.append((x + k * dx, y + k * dy))

            # straight_four _OOOO_
            if all(map(is_stone, range(4))):
                if is_empty(-1) and is_empty(4):
                    fatal.append((x - dx, y - dy))
                    fatal.append((x + 4 * dx, y + 4 * dy))

            # three 1 __OOO__
            if all(map(is_stone, range(3))):
                if all(map(is_empty, [-2, -1, 3, 4])):
                    threat.append((x - dx, y - dy))
                    threat.append((x + 3 * dx, y + 3 * dy))

            # three 2 X_OOO__
            if all(map(is_stone, range(3))):
                if is_block(-2) and all(map(is_empty, [-1, 3, 4])):
                    threat.append((x - dx, y - dy))
                    threat.append((x + 3 * dx, y + 3 * dy))
                    threat.append((x + 4 * dx, y + 4 * dy))

            # broken three _O_OO_
            if all(map(is_stone, [0, 2, 3])) \
                    and all(map(is_empty, [-1, 1, 4])):
                threat.append((x - dx, y - dy))
                threat.append((x + dx, y + dy))
                fatal.append((x + dx, y + dy))
                threat.append((x + 4 * dx, y + 4 * dy))

            # weak three XOOO__
            # weak three XO_OO_
            # weak three XOO_O_
            if is_block(-1) and not any(map(is_block, range(5))) \
                    and sum(map(is_stone, range(4))) == 3:
                ks = filter(is_empty, range(5))
                weak.extend((x + k * dx, y + k * dy) for k in ks)

        return fatal, threat, weak
