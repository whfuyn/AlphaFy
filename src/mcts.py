import os
import random
import numpy as np
import tensorflow as tf
import concurrent.futures
from board import Board
import itertools
from threading import Lock
from pvn import PolicyValueNet
from concurrent.futures import ThreadPoolExecutor
from config import (
    BOARD_SHAPE,
    C_PUCT,
    ALPHA,
    EPSILON,
    NUM_DEFAULT_SEARCH_STEPS,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
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

    def get_Q_plus_U(self, moves):
        U = C_PUCT * self.P[moves] \
            * np.sqrt(self.total_visit or 1) / (1 + self.N[moves])
        return self.Q[moves] + U

    def select_exploratory_move(self):
        xsys = self.board.available_moves
        index = self.get_Q_plus_U(xsys).argmax()
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

    def get_train_data(self):
        return self.board.get_train_data()

    def get_search_probability(self):
        total_move = self.board.total_move
        if total_move <= 0.1 * np.product(BOARD_SHAPE):
            return self.N / (self.total_visit or 1.0)
        else:
            most_visited = np.unravel_index(np.argmax(self.N), BOARD_SHAPE)
            pi = np.zeros(BOARD_SHAPE, dtype=np.float32)
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
                 model_path=DEFAULT_MODEL_PATH,
                 parallel_num=DEFAULT_PARALLEL_NUM,
                 need_data=True):
        self.parallel_num = parallel_num
        self.pool = ThreadPoolExecutor(max_workers=self.parallel_num)
        self.model = PolicyValueNet(model_path=model_path)

        board_path = DEFAULT_DATA_PATH + 'board.np'
        P_path = DEFAULT_DATA_PATH + 'P.np'
        v_path = DEFAULT_DATA_PATH + 'v.np'
        if need_data \
                and os.path.exists(board_path) \
                and os.path.exists(P_path) \
                and os.path.exists(v_path):
            with open(board_path, 'rb') as f_board, \
                    open(P_path, 'rb') as f_P, \
                    open(v_path, 'rb') as f_v:
                board = np.load(f_board)
                v = np.load(f_v)
                P = np.load(f_P)
        else:
            board = np.zeros((0,) + BOARD_SHAPE + (2,), dtype=np.float32)
            P = np.zeros((0,) + BOARD_SHAPE, dtype=np.float32)
            v = np.zeros((0, 1), dtype=np.float32)
        self.train_data = (board, P, v)
        self.nodes = []
        self.new_game()

    def refresh(self):
        shape = (0,) + BOARD_SHAPE
        board = np.zeros(shape + (2,), dtype=np.float32)
        P = np.zeros(shape, dtype=np.float32)
        v = np.zeros((0, 1), dtype=np.float32)
        self.train_data = (board, P, v)
        self.new_game()

# Monte Carlo Tree Search
    def select_leaf(self):
        node = self.root
        while True:
            if node.get_game_state() != 'playing':
                return node
            with node.lock:
                best_move = node.select_exploratory_move()
                # virtual loss, will restore in backup
                if node.origin_P.get(best_move) is None:
                    node.origin_P[best_move] = node.P[best_move]
                node.P[best_move] *= 0.8
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
                lambda node: node.get_game_state() == 'playing',
                leaf_nodes
            ))
            # print('explore: %d' % len(playing_nodes))
            # print('------------------------')
            # print(list(map(id, playing_nodes)))
            # for node in playing_nodes:
            #     node.board.show()
            # print('------------------------')
            self.evaluate(playing_nodes)
            self.pool.map(Node.backup, leaf_nodes)

    def evaluate(self, nodes):
        if len(nodes) != 0:
            boards = [node.board for node in nodes]
            features = np.vstack(
                list([board.get_train_data()] for board in boards)
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
            b = node.get_train_data()
            p = node.get_search_probability()
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
        self.train_data = (bs, Ps, vs)

    def train(self, epochs=5, batch_size=128):
        bs, Ps, vs = self.train_data
        self.model.train(
            x=bs,
            y=[Ps, vs],
            epochs=epochs,
            batch_size=batch_size
        )

# Utility
    def save(self):
        self.model.save()
        with open(DEFAULT_DATA_PATH + 'board.np', 'wb') as f_board, \
                open(DEFAULT_DATA_PATH + 'P.np', 'wb') as f_P, \
                open(DEFAULT_DATA_PATH + 'v.np', 'wb') as f_v:
            board, P, v = self.train_data
            np.save(f_board, board)
            np.save(f_P, P)
            np.save(f_v, v)

    def pick(self, n):
        self.ensure_game_no_end()
        self.explore(n)
        best_move = self.root.select_best_move()
        best_node = self.root.children[best_move]
        if best_node.expanding:
            self.evaluate([best_node])
            best_node.backup()
        return best_move

    def choose(self, n):
        self.ensure_game_no_end()
        move = self.pick(n)
        self.set_move(move)
        return move

    def set_move(self, move):
        self.ensure_game_no_end()
        self.root.expand(move)
        new_root = self.root.children[move]
        if new_root.expanding:
            self.evaluate([new_root])
        new_root.parent = None
        self.nodes.append(self.root)
        self.root = new_root
        if self.get_game_state() != 'playing':
            self.nodes.append(self.root)
            z = -1 if self.get_game_state() == 'lose' else 0
            self.into_train_data(z)

    def self_play(self, count=NUM_DEFAULT_SEARCH_STEPS):
        self.new_game()
        while self.get_game_state() == 'playing':
            move = self.pick(count)
            self.set_move(move)

    def new_game(self):
        self.root = Node(Board(), None)
        self.evaluate([self.root])
        self.nodes.clear()

    def get_game_state(self):
        return self.root.get_game_state()

    def show_board(self):
        self.root.board.show()

    def undo(self):
        pre = self.nodes.pop()
        self.root.parent = pre
        self.root = pre

    def dump_moves(self):
        node = self.root
        moves = []
        while node.parent is not None:
            x, y = node.last_move
            moves.append((x + 1, y + 1))
            node = node.parent
        return moves

    def apply_moves(self, moves):
        for (x, y) in moves:
            self.set_move((x - 1, y - 1))

    def ensure_game_no_end(self):
        if self.get_game_state() != 'playing':
            raise RuntimeError('Game: %s.' % self.get_game_state())
