#!/usr/local/bin/python3

from core.util import Direction, Move, PlayerColor, FieldState
from core.state import GameSettings, GameState
import numpy as np
from time import time
GameSettings.ourColor = PlayerColor.RED
GameSettings.startPlayerColor = PlayerColor.RED
state = GameState()

a = np.full((10, 10), FieldState.EMPTY)
a[6, 2:7] = FieldState.RED
a[5, 7] = FieldState.RED

a[5, 2] = FieldState.BLUE
a[4,1:6] = FieldState.BLUE

a[4, 5] = FieldState.OBSTRUCTED
a[6, 4] = FieldState.OBSTRUCTED
a[4, 3] = FieldState.RED
a[5, 3] = FieldState.RED
a[7, 6] = FieldState.RED
a[9, 5] = FieldState.RED

state.currentPlayerColor = PlayerColor.RED
state.turn = 56
state.board = a

state.printColored()

start_time = time()

our_fishes = state.get_fishes(GameSettings.ourColor)

estimated_rewards = {}  # dict of (move, reward)
for our_fish in our_fishes:
    for our_dir in Direction:
        our_move = Move(our_fish[0], our_fish[1], our_dir)
        # game_state = self.currentGameState.apply(our_move)
        reward, done, game_state = state.estimate_reward(our_move, opponent_did_move=False)
        if game_state is None:
            # this move was invalid
            continue

        estimated_rewards[our_move] = (reward, done, game_state)

estimated_rewards_sorted = sorted(estimated_rewards.items(), key=lambda x: -x[1][0])

estimated_rewards_opponent = {}
time_exceeded = False
for move, (reward, done, game_state) in estimated_rewards_sorted:
    possible_rewards = []
    their_fishes = game_state.get_fishes(GameSettings.ourColor.otherColor())
    for their_fish in their_fishes:
        for their_dir in Direction:
            their_move = Move(their_fish[0], their_fish[1], their_dir)
            next_game_state = game_state.apply(their_move)
            if next_game_state is None:
                # this move was invalid
                continue

            reward, done, _ = state.estimate_reward(next_game_state)
            possible_rewards.append(reward)

            if time() - start_time > 1.4:
                time_exceeded = True
                break
        if time_exceeded:
            break
    if time_exceeded:
        break

    possible_rewards = np.array(possible_rewards)
    estimated_rewards_opponent[move] = (possible_rewards.mean(), possible_rewards.max(), possible_rewards.min())

if len(estimated_rewards_opponent) >= 3:
    estimated_rewards = estimated_rewards_opponent
else:
    for move in estimated_rewards:
        reward = estimated_rewards[move][0]
        estimated_rewards[move] = (reward, reward, reward)

worst_case_move = None
highest_worst_case_reward = None

best_case_move = None
highest_best_case_reward = None

typical_move = None
highest_typical_reward = None
for move, (typical_reward, best_case_reward, worst_case_reward) in estimated_rewards.items():

    if highest_typical_reward is None or typical_reward > highest_typical_reward:
        typical_move = move
        highest_typical_reward = typical_reward

    if highest_best_case_reward is None or best_case_reward > highest_best_case_reward:
        best_case_move = move
        highest_best_case_reward = best_case_reward

    if highest_worst_case_reward is None or worst_case_reward > highest_worst_case_reward:
        worst_case_move = move
        highest_worst_case_reward = worst_case_reward

print(
'''[env] Sending move after {:.3f} seconds. Expected Reward:
    Typical:    {:10.2f} {}
    Best Case:  {:10.2f} {}
    Worst Case: {:10.2f} {}
'''.format(
    time()-start_time,
    highest_typical_reward, typical_move,
    highest_best_case_reward, best_case_move,
    highest_worst_case_reward, worst_case_move
))
