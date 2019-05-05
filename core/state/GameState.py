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

    def __str__(self):
        stringRepresentation = 'GameState(\n' + ' ' * 4
        stringRepresentation += 'currentPlayerColor: ' + str(self.currentPlayerColor) + '\n' + ' ' * 4
        stringRepresentation += 'turn: ' + str(self.turn) + '\n' + ' ' * 4

        stringRepresentation += 'board:'
        if self.board is None:
            stringRepresentation += ' ' + str(None) + '\n'
        else:
            printing = self.board.T
            for idx in range(len(printing)-1, -1, -1):
                row = printing[idx]
                stringRepresentation += '\n' + ' ' * 8
                for item in row:
                    stringRepresentation += str(item.value).rjust(2)
            stringRepresentation += '\n'

        stringRepresentation += ')'
        return stringRepresentation
