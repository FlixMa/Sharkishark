#!/usr/local/bin/python3

import core
import numpy as np

core.state.GameSettings.ourColor = core.util.PlayerColor.RED
state = core.state.GameState()

a = np.full((10, 10), core.util.FieldState.EMPTY)
a[:, 0] = core.util.FieldState.RED
a[:,-1] = core.util.FieldState.RED
a[0, :] = core.util.FieldState.BLUE
a[-1,:] = core.util.FieldState.BLUE
a[0, 0] = core.util.FieldState.EMPTY
a[0, 9] = core.util.FieldState.EMPTY
a[9, 0] = core.util.FieldState.EMPTY
a[9, 9] = core.util.FieldState.EMPTY

a[4, 6] = core.util.FieldState.OBSTRUCTED
a[4, 3] = core.util.FieldState.RED


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
    boolBoard = (board == core.util.FieldState.fromPlayerColor(playerColor))
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

group = findLargestGroup(core.state.GameSettings.ourColor, state.board)
state.printColored(highlight=group)


#
