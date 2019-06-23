from core.state import GameState, GameSettings
from core.util import FieldState, PlayerColor, Direction
import numpy as np

class Move():

    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    def __str__(self):
        return "Move(({}, {}) in direction {})".format(
            self.x, self.y, self.direction)

    def validate(self, game_state, debug=False):
            '''
                Returns a tuple:
                1. flag, if move is valid
                2. destination (x, y) of that move or None if invalid
            '''

            if not isinstance(game_state, GameState.GameState):
                raise ValueError('game_state argument is not of type "GameState". Given: ' + str(type(game_state)))

            if game_state.board is None:
                raise ValueError('No board found.')

            board = game_state.board

            '''
                check if this move is valid

                1. a fish of ours is selected
                2. the fish can move in that direction (not directly next to the bounds)
                3. destination is empty or opponent's fish (not ours and not a kraken)
                4. our fish does not jump over a opponent's fish
            '''

            ourFishFieldState = FieldState.fromPlayerColor(GameSettings.ourColor)

            if not (board[self.x, self.y] == ourFishFieldState):
                if debug:
                    print('Can\'t mind control the opponents fishes :(')
                return False, None

            # count fishes in that row
            #
            # on which axis are we and
            # on that axis - where are we exactly?
            axis = None
            current_position_on_axis = None
            if self.direction == Direction.UP or self.direction == Direction.DOWN:
                axis = board[self.x]
                current_position_on_axis = self.y
            elif self.direction == Direction.LEFT or self.direction == Direction.RIGHT:
                axis = board[:, self.y]
                current_position_on_axis = self.x
            elif self.direction == Direction.DOWN_LEFT or self.direction == Direction.UP_RIGHT:
                axis = board.diagonal(self.y - self.x)
                current_position_on_axis = self.x if self.y > self.x else self.y
            elif self.direction == Direction.UP_LEFT or self.direction == Direction.DOWN_RIGHT:
                flippedX = ((board.shape[0] - 1) - self.x)

                # NOTE: flipud actually flips the board left to right because of the way how we index it
                axis = np.flipud(board).diagonal(self.y - flippedX)

                current_position_on_axis = flippedX if self.y > flippedX else self.y

            if debug:
                print('move', self.direction.name, (self.x, self.y), '-> axis: [ ', end='')
                for item in axis:
                    print(item.name, end=' ')
                print('], idx:', current_position_on_axis)

            num_fishes = ((axis == FieldState.RED) | (axis == FieldState.BLUE)).sum()
            if debug:
                print('-> fishlis:', num_fishes)

            #  where do we wanna go?
            #  NOTE: y is upside down / inverted
            direction_forward = (self.direction in [Direction.UP, Direction.UP_LEFT, Direction.UP_RIGHT, Direction.RIGHT])
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


            dest_x, dest_y = self.x, self.y
            if self.direction == Direction.UP or self.direction == Direction.DOWN:
                dest_y = destination_position_on_axis
            elif self.direction == Direction.LEFT or self.direction == Direction.RIGHT:
                dest_x = destination_position_on_axis
            elif self.direction == Direction.DOWN_LEFT or self.direction == Direction.UP_RIGHT:
                dest_x += num_fishes * (1 if direction_forward else -1)
                dest_y += num_fishes * (1 if direction_forward else -1)
            elif self.direction == Direction.UP_LEFT or self.direction == Direction.DOWN_RIGHT:
                dest_x -= num_fishes * (1 if direction_forward else -1)
                dest_y += num_fishes * (1 if direction_forward else -1)

            return True, (dest_x, dest_y)
