#!/usr/local/bin/python3

import core
import time
import numpy as np
import random

class EvenSimplerGameLogicDelegate(core.logic.GameLogicDelegate):

    def __init__(self):
        self.currentGameState = None

    def onGameStateUpdate(self, gameState):
        super().onGameStateUpdate(gameState)
        self.currentGameState = gameState

    def onMoveRequest(self):
        super().onMoveRequest()
        if self.currentGameState is None:
            print('there is no field')
            return None
        else:
            print('issuing move')


            board = self.currentGameState.board

            myFieldState = core.util.FieldState.fromPlayerColor(core.state.GameSettings.ourColor)

            positions = np.argwhere(board == myFieldState)

            unsortedPositions = random.sample(positions, len(positions))

            for position in unsortedPositions:
                for direction in core.util.Direction:
                    proposedMove = core.util.Move(position[0], position[1], direction)
                    if self.validateMove(proposedMove):
                        print(proposedMove)
                        return proposedMove

            print('all moves are invalid')
            return None


    def validateMove(self, move, gameState=None):
        if not isinstance(move, core.util.Move):
            raise ValueError('move argument is not of type "core.util.Move". Given: ' + str(type(move)))

        if gameState is None:
            gameState = self.currentGameState
        elif not isinstance(gameState, core.state.GameState):
            raise ValueError('gameState argument is not of type "core.state.GameState". Given: ' + str(type(gameState)))

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

        ourFishFieldState = core.util.FieldState.fromPlayerColor(core.state.GameSettings.ourColor)

        if not (board[move.x, move.y] == ourFishFieldState):
            print('Can\'t mind control the opponents fishes :(')
            return False, None

        # count fishes in that row
        #
        # on which axis are we and
        # on that axis - where are we exactly?
        axis = None
        current_position_on_axis = None
        if move.direction == core.util.Direction.UP or move.direction == core.util.Direction.DOWN:
            axis = board[move.x]
            current_position_on_axis = move.y
        elif move.direction == core.util.Direction.LEFT or move.direction == core.util.Direction.RIGHT:
            axis = board[:, move.y]
            current_position_on_axis = move.x
        elif move.direction == core.util.Direction.DOWN_LEFT or move.direction == core.util.Direction.UP_RIGHT:
            axis = board.diagonal(move.y - move.x)
            current_position_on_axis = move.x if move.y > move.x else move.y
        elif move.direction == core.util.Direction.UP_LEFT or move.direction == core.util.Direction.DOWN_RIGHT:
            flippedX = ((board.shape[0] - 1) - move.x)

            # NOTE: flipud actually flips the board left to right because of the way how we index it
            axis = np.flipud(board).diagonal(move.y - flippedX)

            current_position_on_axis = flippedX if move.y > flippedX else move.y

        print('move', move.direction.name, (move.x, move.y), '-> axis: [ ', end='')
        for item in axis:
            print(item.name, end=' ')
        print('], idx:', current_position_on_axis)

        num_fishes = ((axis == core.util.FieldState.RED) | (axis == core.util.FieldState.BLUE)).sum()
        print('-> fishlis:', num_fishes)

        #  where do we wanna go?
        #  NOTE: y is upside down / inverted
        direction_forward = (move.direction in [core.util.Direction.UP, core.util.Direction.UP_LEFT, core.util.Direction.UP_RIGHT, core.util.Direction.RIGHT])
        destination_position_on_axis = (current_position_on_axis + num_fishes) if direction_forward else (current_position_on_axis - num_fishes)
        print('direction_forward:', direction_forward)
        print('destination:', destination_position_on_axis)

        # check for bounds
        if destination_position_on_axis < 0 or destination_position_on_axis >= axis.size:
            print('Exceeding bounds. %d of %d' % (destination_position_on_axis, axis.size))
            return False, None

        # what type is that destination field?
        destinationFieldState = axis[destination_position_on_axis]
        if destinationFieldState == core.util.FieldState.OBSTRUCTED or destinationFieldState == ourFishFieldState:
            print('Destination is obstructed or own fish:', destinationFieldState)
            return False, None

        # is an opponents fish in between(! excluding the destiantion !)?
        opponentsFieldState = core.util.FieldState.RED if ourFishFieldState == core.util.FieldState.BLUE else core.util.FieldState.BLUE
        for idx in range(current_position_on_axis, destination_position_on_axis, 1 if direction_forward else -1):
            if axis[idx] == opponentsFieldState:
                print('Can\'t jump over opponents fish.')
                return False, None


        dest_x, dest_y = move.x, move.y
        if move.direction == core.util.Direction.UP or move.direction == core.util.Direction.DOWN:
            dest_y = destination_position_on_axis
        elif move.direction == core.util.Direction.LEFT or move.direction == core.util.Direction.RIGHT:
            dest_x = destination_position_on_axis
        elif move.direction == core.util.Direction.DOWN_LEFT or move.direction == core.util.Direction.UP_RIGHT:
            dest_x += num_fishes * (1 if direction_forward else -1)
            dest_y += num_fishes * (1 if direction_forward else -1)
        elif move.direction == core.util.Direction.UP_LEFT or move.direction == core.util.Direction.DOWN_RIGHT:
            dest_x -= num_fishes * (1 if direction_forward else -1)
            dest_y += num_fishes * (1 if direction_forward else -1)

        return True, (dest_x, dest_y)




'''
gameLogic = EvenSimplerGameLogicDelegate()
gameClient = core.communication.GameClient('127.0.0.1', 13050, gameLogic)

gameClient.start()
gameClient.join()

while not gameClient.is_stopped():
    try:
        time.sleep(100)
    except:
        gameClient.stop()
'''
#time.sleep(15)
#gameClient.stop()

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

core.state.GameSettings.ourColor = core.util.PlayerColor.RED
delegate = EvenSimplerGameLogicDelegate()
delegate.currentGameState = state


#state.printColored(highlight=[(4, 0), (6, 2)])
#delegate.validateMove(core.util.Move(4, 0, core.util.Direction.UP_LEFT))



for dir in core.util.Direction:
    print('trying to move in', dir.name)

    valid, destination = delegate.validateMove(core.util.Move(4, 3, dir))
    state.printColored(highlight=[destination])

    print('\n' + '- ' * 10 + '\n')




#
