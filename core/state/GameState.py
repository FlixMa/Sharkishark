import core

class GameState():
    def __init__(self):
        self.currentPlayerColor = None
        self.turn = None
        self.board = None

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
                        elif item == core.util.FieldState.RED:
                            temp = TerminalColor.RED.wrap(temp)
                        elif item == core.util.FieldState.BLUE:
                            temp = TerminalColor.CYAN.wrap(temp)
                        elif item == core.util.FieldState.OBSTRUCTED:
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
