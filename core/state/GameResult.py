from enum import Enum

class GameResult(Enum):
    LOST    = 0
    TIED    = 1
    WON     = 2

    @classmethod
    def fromInt(cls, result):
        if not isinstance(result, int):
            return None

        if result == 0:
            return cls.LOST
        elif result == 1:
            return cls.TIED
        elif result == 2:
            return cls.WON
        else:
            return None
