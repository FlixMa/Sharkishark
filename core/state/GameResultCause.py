from enum import Enum

class GameResultCause(Enum):
    REGULAR         = 0
    LEFT            = 1
    RULE_VIOLATION  = 2
    SOFT_TIMEOUT    = 3
    HARD_TIMEOUT    = 4

    @classmethod
    def fromString(cls, causeString):
        if not isinstance(causeString, str):
            return None

        causeString = causeString.upper()
        if causeString == 'REGULAR':
            return cls.REGULAR
        elif causeString == 'LEFT':
            return cls.LEFT
        elif causeString == 'RULE_VIOLATION':
            return cls.RULE_VIOLATION
        elif causeString == 'SOFT_TIMEOUT':
            return cls.SOFT_TIMEOUT
        elif causeString == 'HARD_TIMEOUT':
            return cls.HARD_TIMEOUT
        else:
            return None
