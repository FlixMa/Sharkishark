from core.util import FieldState, Move
from core.state import GameSettings
import numpy as np

class GameState():
    def __init__(self):
        self.currentPlayerColor = None
        self.turn = None
        self.board = None

    def calc_mean_distance_using_median_center(self, player_color):
        fishes = np.argwhere(self.board == FieldState.fromPlayerColor(player_color))
        median_coordinate = np.percentile(
            fishes,
            50,
            axis=0,
            interpolation='nearest'
        )

        squared_error = 0
        for fish_coord in fishes:
            squared_error += np.square(median_coordinate - fish_coord).sum()
        return squared_error / len(fishes)

    def find_groups(self, player_color):
        boolBoard = (self.board == FieldState.fromPlayerColor(player_color))
        fishesToConsider = np.argwhere(boolBoard)
        groups = []

        # calculate group for each starting position
        while len(fishesToConsider) > 0:
            fish = fishesToConsider[0]

            # get neighborhood of chosen fish and iteratively do a flood-fill
            fish_neighborhood = GameState.neighbors(fish[0], fish[1], boolBoard)
            i = 0
            while i < len(fish_neighborhood):
                neighbor = fish_neighborhood[i]

                its_neighborhood = GameState.neighbors(neighbor[0], neighbor[1], boolBoard)
                concatenated = np.concatenate((fish_neighborhood, its_neighborhood))
                fish_neighborhood = np.unique(concatenated, axis=0)
                i += 1

            groups.append(fish_neighborhood)

            # remove all fishes in this group from the fishes to consider next round
            remainingMask = ~(np.isin(list(map(lambda a: a[0] * 10 + a[1], fishesToConsider)),
                                      list(map(lambda a: a[0] * 10 + a[1], fish_neighborhood))))
            fishesToConsider = fishesToConsider[remainingMask]

        return groups

    def get_biggest_group(self, player_color):
        groups = self.find_groups(player_color)
        largestGroup = None
        largestSize = 0
        for group in groups:
            size = len(group)
            if largestSize < size:
                largestSize = size
                largestGroup = group

        return largestGroup

    def apply(self, move):
        # next board?
        is_valid, destination = move.validate(self)
        if not is_valid:
            return None

        next_game_state = GameState.copy(self)

        if self.board is not None:
            next_game_state.board[move.x, move.y] = FieldState.EMPTY
            next_game_state.board[destination[0], destination[1]] = FieldState.fromPlayerColor(GameSettings.ourColor)

        if self.currentPlayerColor is not None:
            next_game_state.currentPlayerColor = self.currentPlayerColor.otherColor()

        if self.turn is not None:
            next_game_state.turn += 1

        return next_game_state

    def number_of_fishes(self, player_color):
        return (self.board == FieldState.fromPlayerColor(player_color)).sum()

    @staticmethod
    def neighbors(x, y, boolBoard):

        xmin = max(0, x - 1)
        xmax = min(x + 2, boolBoard.shape[0])
        ymin = max(0, y - 1)
        ymax = min(y + 2, boolBoard.shape[1])

        neighborhood = boolBoard[xmin:xmax, ymin:ymax]
        # print('umfeld')
        # print(neighborhood)

        neighbors = np.argwhere(neighborhood)

        neighbors += np.array([xmin, ymin])

        return neighbors


    def estimate_reward(self, argument):
        next_game_state = None
        if isinstance(argument, GameState):
            next_game_state = argument
        elif isinstance(argument, Move):
            # apply move to the board and check if it's valid
            next_game_state = self.apply(argument)
            if next_game_state is None:
                return -100.0, True
        else:
            raise ValueError('Can\'t determine next game state to compare with. Argument is of invalid type. Expected GameState or Move. Got: ' + str(type(argument)))

        our_current_mean_distance = self.calc_mean_distance_using_median_center(GameSettings.ourColor)
        our_next_mean_distance = next_game_state.calc_mean_distance_using_median_center(GameSettings.ourColor)
        our_distance_increase = our_next_mean_distance - our_current_mean_distance

        their_current_mean_distance = self.calc_mean_distance_using_median_center(GameSettings.ourColor.otherColor())
        their_next_mean_distance = next_game_state.calc_mean_distance_using_median_center(GameSettings.ourColor.otherColor())
        their_distance_increase = their_next_mean_distance - their_current_mean_distance

        our_current_biggest_group = self.get_biggest_group(GameSettings.ourColor)
        our_next_biggest_group = next_game_state.get_biggest_group(GameSettings.ourColor)
        our_group_increase = len(our_next_biggest_group) - len(our_current_biggest_group)

        their_current_biggest_group = self.get_biggest_group(GameSettings.ourColor.otherColor())
        their_next_biggest_group = next_game_state.get_biggest_group(GameSettings.ourColor.otherColor())
        their_group_increase = len(their_next_biggest_group) - len(their_current_biggest_group)

        our_current_num_fishes = self.number_of_fishes(GameSettings.ourColor)
        our_next_num_fishes = next_game_state.number_of_fishes(GameSettings.ourColor)
        their_current_num_fishes = self.number_of_fishes(GameSettings.ourColor.otherColor())
        their_next_num_fishes = next_game_state.number_of_fishes(GameSettings.ourColor.otherColor())

        # the more fishes are united the closer we are to winning
        our_group_union_fraction = float(len(our_next_biggest_group)) / our_next_num_fishes
        their_group_union_fraction = float(len(their_next_biggest_group)) / their_next_num_fishes
        group_is_swarmier_reward = (10.0 if our_group_union_fraction > their_group_union_fraction else -10.0)

        # if its getting to a draw situation, its just necessary to have more fishes unioned than the other player
        group_has_more_fishes_reward = (10.0 if len(our_next_biggest_group) > len(their_next_biggest_group) else -10.0)

        game_progress = (self.turn / 60.0) ** 5
        group_reward = (1 - game_progress) * group_is_swarmier_reward + game_progress * group_has_more_fishes_reward

        distance_reward = their_distance_increase - our_distance_increase
        group_increase_reward = our_group_increase - their_group_increase

        # TODO rewards for swarm disrupted, Strafe for fish eaten, look at opponent's reward
        return (
            group_reward + distance_reward + group_increase_reward,
            next_game_state.turn >= 60
        )


    @classmethod
    def copy(cls, other):
        if isinstance(other, GameState):
            state = cls()
            state.currentPlayerColor = other.currentPlayerColor
            state.turn = other.turn
            if other.board is not None:
                state.board = other.board.copy()
            return state

        raise ValueError('other is not of type GameState. Given: %s' % type(other))

    def __eq__(self, other):
        if isinstance(other, GameState):
            return self.currentPlayerColor == other.currentPlayerColor\
                and self.turn == other.turn\
                and self.board == other.board

        raise ValueError('other is not of type GameState. Given: %s' % type(other))

    def __ne__(self, other):
        if isinstance(other, GameState):
            return self.currentPlayerColor != other.currentPlayerColor\
                or self.turn != other.turn\
                or self.board != other.board

        raise ValueError('other is not of type GameState. Given: %s' % type(other))

    def __str__(self, colored=False, highlight=[]):
        stringRepresentation = 'GameState(\n' + ' ' * 4
        stringRepresentation += 'currentPlayerColor: ' + str(self.currentPlayerColor) + '\n' + ' ' * 4
        stringRepresentation += 'turn: ' + str(self.turn) + '\n' + ' ' * 4

        stringRepresentation += 'board:'
        if self.board is None:
            stringRepresentation += ' ' + str(None) + '\n'
        else:
            printing = self.board.T
            for y in range(len(printing)-1, -1, -1):
                row = printing[y]
                stringRepresentation += '\n' + ' ' * 8
                for x, item in enumerate(row):
                    temp = str(item.value)

                    if colored:
                        if tuple([x, y]) in list(map(lambda a: tuple((a[0], a[1])), highlight)):
                            temp = TerminalColor.BG_GREEN.wrap(temp)
                        elif item == FieldState.RED:
                            temp = TerminalColor.RED.wrap(temp)
                        elif item == FieldState.BLUE:
                            temp = TerminalColor.CYAN.wrap(temp)
                        elif item == FieldState.OBSTRUCTED:
                            temp = TerminalColor.GREEN.wrap(temp)

                    stringRepresentation += ' ' + temp

            stringRepresentation += '\n'

        stringRepresentation += ')'
        return stringRepresentation

    def printColored(self, highlight=[]):
        print(self.__str__(colored=True, highlight=highlight))

from enum import Enum

class TerminalColor(Enum):
    RESET = "\033[0;0m"
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    BG_GREEN = "\033[0;103m"
    def wrap(self, text):
        return self.value + str(text) + TerminalColor.RESET.value
