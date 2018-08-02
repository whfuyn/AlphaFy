import os
import random
import itertools
import numpy as np
from board import Board
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from config import (
    BOARD_SHAPE,
    C_PUCT,
    ALPHA,
    EPSILON,
    VIRTUAL_DISCOUNT,
    DEFAULT_PARALLEL_NUM
)


class Node:
    def __init__(self, board, parent):
        self.board = board
        self.parent = parent
        if self.board.game_state != 'playing':
            self.P = np.zeros(BOARD_SHAPE, dtype=np.float32)
            self.v = -1 if self.board.game_state == 'lose' else 0
            self.expanding = False
        else:
            self.expanding = True
        self.W = np.zeros(BOARD_SHAPE, dtype=np.float32)
        self.Q = np.zeros(BOARD_SHAPE, dtype=np.float32)
        self.N = np.zeros(BOARD_SHAPE, dtype=np.int32)
        self.total_visit = 0
        self.children = np.full(BOARD_SHAPE, None)
        self.origin_P = dict()
        self.lock = Lock()

    def get_exploratory_value(self, moves):
        U = C_PUCT * self.P[moves] \
            * np.sqrt(self.total_visit or 1) / (1 + self.N[moves])
        return self.Q[moves] + U

    def select_exploratory_move(self):
        xsys = self.board.available_moves
        index = self.get_exploratory_value(xsys).argmax()
        move = (xsys[0][index], xsys[1][index])
        self.expand(move)
        return move

    def select_best_move(self):
        pi = self.get_search_probability()
        index = np.random.choice(np.product(BOARD_SHAPE), p=pi.flatten())
        move = np.unravel_index(index, BOARD_SHAPE)
        return move

    def expand(self, move):
        if self.children[move] is None:
            new_board = self.board.put(move)
            new_node = Node(new_board, self)
            self.children[move] = new_node

    def update(self, P, v):
        self.P = self.add_dirichlet_noise(P)
        self.v = v
        self.expanding = False

    def backup(self):
        node = self
        pa = node.parent
        v = self.v
        while pa is not None:
            with pa.lock:
                last_move = node.board.last_move
                pa.total_visit += 1
                pa.N[last_move] += 1
                pa.W[last_move] += (-1) * v
                pa.Q[last_move] = pa.W[last_move] / pa.N[last_move]
                for (move, p) in pa.origin_P.items():
                    pa.P[move] = p
            node = pa
            pa = node.parent
            v = (-1) * v

    def get_game_state(self):
        return self.board.game_state

    def is_game_end(self):
        return self.get_game_state() != 'playing'

    def get_board_data(self):
        return self.board.get_board_data()

    def get_search_probability(self):
        if self.total_visit == 0:
            return np.ones(BOARD_SHAPE) / np.product(BOARD_SHAPE)
        total_move = self.board.total_move
        if total_move <= 0.1 * np.product(BOARD_SHAPE):
            return self.N / self.total_visit
        else:
            most_visited = np.unravel_index(np.argmax(self.N), BOARD_SHAPE)
            pi = np.zeros(BOARD_SHAPE)
            pi[most_visited] = 1.0
            return pi

    def add_dirichlet_noise(self, P):
        moves = list(zip(*self.board.available_moves))
        noise = np.random.gamma(ALPHA, size=len(moves))
        noise /= np.sum(noise)
        for (i, move) in enumerate(moves):
            P[move] = (1 - EPSILON) * P[move] + EPSILON * noise[i]
        return P


class MonteCarloTreeSearch:
    def __init__(self,
                 model,
                 parallel_num=DEFAULT_PARALLEL_NUM):
        self.model = model
        self.parallel_num = parallel_num
        self.pool = ThreadPoolExecutor(max_workers=self.parallel_num)

        board = np.zeros((0,) + BOARD_SHAPE + (2,), dtype=np.float32)
        P = np.zeros((0,) + BOARD_SHAPE, dtype=np.float32)
        v = np.zeros((0, 1), dtype=np.float32)

        # np.array([board, P, v], dtype=object) doesn't work.
        # but np.array([board, P, v, 'emm'], dtype=object) works, interesting.
        self.train_data = np.empty(3, dtype=object)
        self.train_data[:] = board, P, v

        self.nodes = []
        self.new_game()

    def load_data(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                bs, Ps, vs = np.load(f)
                self.train_data[:] = bs, Ps, vs
            print("mcts: Data loaded. Path: \"{}\"".format(path))
        else:
            print('mcts: Loading data failed, data file is not found.')

# Monte Carlo Tree Search
    def select_leaf(self):
        node = self.root
        while True:
            if node.is_game_end():
                return node
            with node.lock:
                best_move = node.select_exploratory_move()
                # Virtual discount, will be restored in backup
                if node.origin_P.get(best_move) is None:
                    node.origin_P[best_move] = node.P[best_move]
                node.P[best_move] *= VIRTUAL_DISCOUNT
                next_node = node.children[best_move]
                if next_node.expanding:
                    return next_node
                node = next_node

    def explore(self, n):
        for i in range(int(np.ceil(n / self.parallel_num))):
            leaf_nodes = self.pool.map(
                lambda _: self.select_leaf(), range(self.parallel_num)
            )
            leaf_nodes = list(set(leaf_nodes))
            playing_nodes = list(filter(
                lambda node: node.is_game_end() is False,
                leaf_nodes
            ))
            self.evaluate(playing_nodes)
            self.pool.map(Node.backup, leaf_nodes)

    def evaluate(self, nodes):
        if len(nodes) != 0:
            features = np.vstack(
                list([node.get_board_data()] for node in nodes)
            )
            Ps, vs = self.model.predict(features)
            list(itertools.starmap(
                lambda node, P, v: node.update(P, v), zip(nodes, Ps, vs)
            ))

# Network Training
    def into_train_data(self, z):
        bs = []
        Ps = []
        vs = []
        for node in reversed(self.nodes):
            b = node.get_board_data()
            p = node.get_search_probability()
            # Data augmentation
            for _ in range(4):
                r = (1, -1)
                for rx, ry in itertools.product(r, r):
                    bs.append([b[::rx, ::ry]])
                    Ps.append([p[::rx, ::ry]])
                    vs.append([z])
                b = np.rot90(b)
                p = np.rot90(p)
            z *= -1
        b, P, v = self.train_data
        bs.append(b)
        Ps.append(P)
        vs.append(v)
        bs = np.vstack(bs)
        Ps = np.vstack(Ps)
        vs = np.vstack(vs)
        self.train_data = np.empty(3, dtype=object)
        self.train_data[:] = bs, Ps, vs

    def train(self, epochs=5, batch_size=128):
        bs, Ps, vs = self.train_data
        self.model.train(
            x=bs,
            y=[Ps, vs],
            epochs=epochs,
            batch_size=batch_size
        )

# Utility
    def save_data(self, path, override=False):
        if os.path.exists(path) and override is False:
            print(
                "mcts: Saving training data failed, file already existed.",
                "Try to call with override=True."
            )
        else:
            with open(path, 'wb') as f:
                np.save(f, self.train_data)
            print('mcts: Data saved. Path: "{}"'.format(path))

    def save_model(self, path, override=False):
        if os.path.exists(path) and override is False:
            print(
                "mcts: Saving model failed, file already existed.",
                "Try to call with override=True."
            )
        else:
            self.model.save(path)

    def pick(self, n):
        self.ensure_game_no_end()
        self.explore(n)
        best_move = self.root.select_best_move()
        best_node = self.root.children[best_move]
        if best_node.expanding:
            self.evaluate([best_node])
            best_node.backup()
        return best_move

    def set_move(self, move):
        self.ensure_game_no_end()
        self.root.expand(move)
        new_root = self.root.children[move]
        if new_root.expanding:
            self.evaluate([new_root])
        new_root.parent = None
        self.nodes.append(self.root)
        self.root = new_root
        if self.is_game_end():
            z = -1 if self.get_game_state() == 'lose' else 0
            self.into_train_data(z)

    def new_game(self):
        self.root = Node(Board(), None)
        self.evaluate([self.root])
        self.nodes.clear()

    def get_game_state(self):
        return self.root.get_game_state()

    def is_game_end(self):
        return self.get_game_state() != 'playing'

    def show(self):
        print(self.root.board)

    def undo(self):
        pre = self.nodes.pop()
        self.root.parent = pre
        self.root = pre

    def dump_moves(self):
        moves = []
        for node in self.nodes[1:]:
            moves.append(node.last_move)
        return moves

    def apply_moves(self, moves):
        for move in moves:
            self.set_move(move)

    def ensure_game_no_end(self):
        if self.is_game_end():
            raise RuntimeError('Game: %s.' % self.get_game_state())
