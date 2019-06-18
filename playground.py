#!/usr/local/bin/python3

from core.util import Direction, Move, PlayerColor, FieldState
from core.state import GameSettings, GameState
import numpy as np

GameSettings.ourColor = PlayerColor.RED
state = GameState()

a = np.full((10, 10), FieldState.EMPTY)
a[1:-1, 0] = FieldState.RED
a[1:-1,-1] = FieldState.RED
a[0, 1:-1] = FieldState.BLUE
a[-1,1:-1] = FieldState.BLUE

a[4, 6] = FieldState.OBSTRUCTED
a[7, 3] = FieldState.OBSTRUCTED
#a[4, 3] = FieldState.RED


state.board = a
#state.printColored()
#print(state.board)

#state.board = np.rot90(state.board)
state.printColored()
#print(state.board)

def neighbors(x, y, boolBoard):


    xmin = max(0, x-1)
    xmax = min(x+2, boolBoard.shape[0])
    ymin = max(0, y-1)
    ymax = min(y+2, boolBoard.shape[1])

    neighborhood = boolBoard[xmin:xmax, ymin:ymax]
    #print('umfeld')
    #print(neighborhood)

    neighbors = np.argwhere(neighborhood)

    neighbors += np.array([xmin, ymin])


    return neighbors

def findGroups(playerColor, board):
    boolBoard = (board == FieldState.fromPlayerColor(playerColor))
    fishesToConsider = np.argwhere(boolBoard)
    groups = []

    # calculate group for each starting position
    while len(fishesToConsider) > 0:
        fish = fishesToConsider[0]

        # get neighborhood of chosen fish and iteratively do a flood-fill
        fish_neighborhood = neighbors(fish[0], fish[1], boolBoard)
        i = 0
        while i < len(fish_neighborhood):
            neighbor = fish_neighborhood[i]

            its_neighborhood = neighbors(neighbor[0], neighbor[1], boolBoard)
            concatenated = np.concatenate((fish_neighborhood, its_neighborhood))
            fish_neighborhood = np.unique(concatenated, axis=0)
            i += 1

        groups.append(fish_neighborhood)

        # remove all fishes in this group from the fishes to consider next round
        remainingMask = ~(np.isin(list(map(lambda a: a[0] * 10 + a[1], fishesToConsider)), list(map(lambda a: a[0] * 10 + a[1], fish_neighborhood))))
        fishesToConsider = fishesToConsider[remainingMask]

    return groups



def findLargestGroup(playerColor, board):

    groups = findGroups(playerColor, board)
    largestGroup = None
    largestSize = 0
    for group in groups:
        size = len(group)
        if largestSize < size:
            largestSize = size
            largestGroup = group

    return largestGroup
'''
group = findLargestGroup(GameSettings.ourColor, state.board)
state.printColored(highlight=group)
'''

def validate_move(move, gameState=None, debug=False):
    '''
        Returns a tuple:
        1. flag, if move is valid
        2. destination (x, y) of that move or None if invalid
    '''

    if not isinstance(move, Move):
        raise ValueError('move argument is not of type "Move". Given: ' + str(type(move)))

    if gameState is None:
        raise ValueError('gameState argument is None')
    elif not isinstance(gameState, GameState):
        raise ValueError('gameState argument is not of type "GameState". Given: ' + str(type(gameState)))

    if gameState is None:
        raise ValueError('No gameState found.')
    elif gameState.board is None:
        raise ValueError('No board found.')

    board = gameState.board

    '''
        check if this move is valid

        1. a fish of ours is selected
        2. the fish can move in that direction (not directly next to the bounds)
        3. destination is empty or opponent's fish (not ours and not a kraken)
        4. our fish does not jump over a opponent's fish
    '''

    ourFishFieldState = FieldState.fromPlayerColor(GameSettings.ourColor)

    if not (board[move.x, move.y] == ourFishFieldState):
        if debug:
            print('Can\'t mind control the opponents fishes :(')
        return False, None

    # count fishes in that row
    #
    # on which axis are we and
    # on that axis - where are we exactly?
    axis = None
    current_position_on_axis = None
    if move.direction == Direction.UP or move.direction == Direction.DOWN:
        axis = board[move.x]
        current_position_on_axis = move.y
    elif move.direction == Direction.LEFT or move.direction == Direction.RIGHT:
        axis = board[:, move.y]
        current_position_on_axis = move.x
    elif move.direction == Direction.DOWN_LEFT or move.direction == Direction.UP_RIGHT:
        axis = board.diagonal(move.y - move.x)
        current_position_on_axis = move.x if move.y > move.x else move.y
    elif move.direction == Direction.UP_LEFT or move.direction == Direction.DOWN_RIGHT:
        flippedX = ((board.shape[0] - 1) - move.x)

        # NOTE: flipud actually flips the board left to right because of the way how we index it
        axis = np.flipud(board).diagonal(move.y - flippedX)

        current_position_on_axis = flippedX if move.y > flippedX else move.y

    if debug:
        print('move', move.direction.name, (move.x, move.y), '-> axis: [ ', end='')
        for item in axis:
            print(item.name, end=' ')
        print('], idx:', current_position_on_axis)

    num_fishes = ((axis == FieldState.RED) | (axis == FieldState.BLUE)).sum()
    if debug:
        print('-> fishlis:', num_fishes)

    #  where do we wanna go?
    #  NOTE: y is upside down / inverted
    direction_forward = (move.direction in [Direction.UP, Direction.UP_LEFT, Direction.UP_RIGHT, Direction.RIGHT])
    destination_position_on_axis = (current_position_on_axis + num_fishes) if direction_forward else (current_position_on_axis - num_fishes)
    if debug:
        print('direction_forward:', direction_forward)
        print('destination:', destination_position_on_axis)

    # check for bounds
    if destination_position_on_axis < 0 or destination_position_on_axis >= axis.size:
        if debug:
            print('Exceeding bounds. %d of %d' % (destination_position_on_axis, axis.size))
        return False, None

    # what type is that destination field?
    destinationFieldState = axis[destination_position_on_axis]
    if destinationFieldState == FieldState.OBSTRUCTED or destinationFieldState == ourFishFieldState:
        if debug:
            print('Destination is obstructed or own fish:', destinationFieldState)
        return False, None

    # is an opponents fish in between(! excluding the destiantion !)?
    opponentsFieldState = FieldState.RED if ourFishFieldState == FieldState.BLUE else FieldState.BLUE
    for idx in range(current_position_on_axis, destination_position_on_axis, 1 if direction_forward else -1):
        if axis[idx] == opponentsFieldState:
            if debug:
                print('Can\'t jump over opponents fish.')
            return False, None


    dest_x, dest_y = move.x, move.y
    if move.direction == Direction.UP or move.direction == Direction.DOWN:
        dest_y = destination_position_on_axis
    elif move.direction == Direction.LEFT or move.direction == Direction.RIGHT:
        dest_x = destination_position_on_axis
    elif move.direction == Direction.DOWN_LEFT or move.direction == Direction.UP_RIGHT:
        dest_x += num_fishes * (1 if direction_forward else -1)
        dest_y += num_fishes * (1 if direction_forward else -1)
    elif move.direction == Direction.UP_LEFT or move.direction == Direction.DOWN_RIGHT:
        dest_x -= num_fishes * (1 if direction_forward else -1)
        dest_y += num_fishes * (1 if direction_forward else -1)

    return True, (dest_x, dest_y)

import time
startTime = time.time()

possible_moves = np.zeros((10 * 10 * (8 + 2)), dtype=np.bool)
positions = np.argwhere(state.board == FieldState.fromPlayerColor(GameSettings.ourColor))

dir_enumerated = list(enumerate(Direction))
for x, y in positions:
    for i, dir in dir_enumerated:
        move = Move(x, y, dir)
        idx = (x + (y * 10)) * 8 + i
        assert(idx < 800)
        possible_moves[idx] = validate_move(move, state)[0]

possible_moves[-200:-100] = (state.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor())).flatten()
possible_moves[-100:] = (state.board == FieldState.OBSTRUCTED).flatten()

endTime = time.time()

print((endTime - startTime) * 1000, 'ms')
for i in range(0, 1000, 100):
    print(possible_moves[i:i+100])
#
