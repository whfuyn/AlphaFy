#!/usr/bin/python3
import os
import sys
import numpy as np
from time import time
from mcts import MonteCarloTreeSearch

mcts = MonteCarloTreeSearch(need_data=False)


def vs(player1, player2, steps1=1024, steps2=1024, show_board=True):
    player1.new_game()
    player2.new_game()
    count = 0
    if show_board:
        player1.show_board()
    while True:
        move1 = player1.choose(steps1)
        player2.set_move(move1)
        if show_board:
            player2.show_board()
        count += 1
        if player2.get_game_state() != 'playing':
            break
        move2 = player2.choose(steps2)
        player1.set_move(move2)
        if show_board:
            player1.show_board()
        count += 1
        if player1.get_game_state() != 'playing':
            break
    if not show_board:
        player2.show_board()
    if count == 81:
        print('{} vs {} | DRAW'.format('player1', 'player2'))
    elif count % 2 == 1:
        print('{} vs {} | {} WIN'.format('player1', 'player2', 'player1'))
    else:
        print('{} vs {} | {} WIN'.format('player1', 'player2', 'player2'))


def play(n, need_restart=True):
    if need_restart:
        mcts.new_game()
    while True:
        x, y = mcts.pick(n)
        print(mcts.root.N)
        print(mcts.root.v)
        print(((x+1, y+1)))
        mcts.set_move((x, y))
        mcts.show_board()
        if mcts.root.get_game_state() != 'playing':
            break
        x, y = eval(input('move:'))
        mcts.set_move((x-1, y-1))
        mcts.show_board()
        if mcts.root.get_game_state() != 'playing':
            break


def play_first(n, need_restart=True):
    if need_restart:
        mcts.new_game()
    while True:
        x, y = eval(input('move:'))
        mcts.set_move((x-1, y-1))
        mcts.show_board()
        if mcts.root.get_game_state() != 'playing':
            break
        x, y = mcts.pick(n)
        print(mcts.root.N)
        print(mcts.root.v)
        print(((x+1, y+1)))
        mcts.set_move((x, y))
        mcts.show_board()
        if mcts.root.get_game_state() != 'playing':
            break


def self_play(n, need_restart=True):
    if need_restart:
        mcts.new_game()
    cnt = 0
    while True:
        x, y = mcts.pick(n)
        print(mcts.root.P)
        print(mcts.root.Q)
        print(mcts.root.N)
        print(mcts.root.v)
        print(((x+1, y+1)))
        mcts.set_move((x, y))
        mcts.show_board()
        cnt += 1
        if mcts.root.get_game_state() != 'playing':
            break
        x, y = mcts.pick(n)
        print(mcts.root.P)
        print(mcts.root.Q)
        print(mcts.root.N)
        print(mcts.root.v)
        print(((x+1, y+1)))
        mcts.set_move((x, y))
        mcts.show_board()
        cnt += 1
        if mcts.root.get_game_state() != 'playing':
            break
    print('%d moves.' % cnt)


def pplay(n, need_restart=True):
    if need_restart:
        mcts.new_game()
    cnt = 0
    while True:
        x, y = eval(input('move:'))
        mcts.set_move((x-1, y-1))
        print(mcts.show_board())
        cnt += 1
        if mcts.root.get_game_state() != 'playing':
            break
        x, y = eval(input('move:'))
        mcts.set_move((x-1, y-1))
        print(mcts.show_board())
        cnt += 1
        if mcts.root.get_game_state() != 'playing':
            break
    print('%d moves.' % cnt) 


def run(n=256):
    count = 1
    total_time = 0

    while True:
        print('======================')
        print('Round {}'.format(count))
        print('Self-playing...')
        sys.stdout.flush()
        t = time()
        mcts.self_play(n)
        t = int(time() - t)
        total_time += t
        print('Finished in {} s.'.format(t))
        sys.stdout.flush()
        count += 1
        if count % 5 == 0:
            # print('Start training...')
            # t = time()
            # mcts.train(epochs=1, batch_size=256)
            # print('Saving model...')
            mcts.save()
            # t = int(time() - t)
            # total_time += t
            # print('Finished in {} s'.format(t))
            print('Model saved.')
            print('Total time elapsed: {} s'.format(total_time))
            sys.stdout.flush()


def load():
    paths = ['./model/%d/' % i for i in range(13, 16)]
    boards = np.vstack([np.load(path + 'board.np') for path in paths])
    Ps = np.vstack([np.load(path + 'P.np') for path in paths])
    vs = np.vstack([np.load(path + 'v.np') for path in paths])
    mcts.train_data = (boards, Ps, vs)


def train():
    mcts.train(epochs=1, batch_size=512)

# run(512)
