from enum import Enum

class GameResultCause(Enum):
    REGULAR         = 0
    RULE_VIOLATION  = 1
    SOFT_TIMEOUT    = 2
    HARD_TIMEOUT    = 3
