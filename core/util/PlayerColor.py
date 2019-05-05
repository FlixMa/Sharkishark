from enum import Enum

class PlayerColor(Enum):
    RED = 1
    BLUE = 2

    @classmethod
    def fromString(cls, playerColorString):
        if not isinstance(playerColorString, str):
            return None

        playerColorString = playerColorString.upper()
        if playerColorString == 'RED':
            return cls.RED
        elif playerColorString == 'BLUE':
            return cls.BLUE
        else:
            return None


    @classmethod
    def fromFieldState(cls, fieldState):
        from .FieldState import FieldState # NOTE: this prevents circular dependency errors on startup

        if fieldState == FieldState.RED:
            return cls.RED
        elif fieldState == FieldState.BLUE:
            return cls.BLUE
        else:
            return None
