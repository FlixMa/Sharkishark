from enum import Enum

class FieldState(Enum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    OBSTRUCTED = 3

    @classmethod
    def fromString(cls, fieldStateString):
        if not isinstance(fieldStateString, str):
            return None

        fieldStateString = fieldStateString.upper()
        if fieldStateString == 'EMPTY':
            return cls.EMPTY
        elif fieldStateString == 'RED':
            return cls.RED
        elif fieldStateString == 'BLUE':
            return cls.BLUE
        elif fieldStateString == 'OBSTRUCTED':
            return cls.OBSTRUCTED
        else:
            return None

    @classmethod
    def fromPlayerColor(cls, playerColor):
        from .PlayerColor import PlayerColor # NOTE: this prevents circular dependency errors on startup

        if playerColor == PlayerColor.RED:
            return cls.RED
        elif playerColor == PlayerColor.BLUE:
            return cls.BLUE
