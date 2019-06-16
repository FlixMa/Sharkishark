from enum import Enum


class Direction(Enum):
    UP = 'UP'
    UP_RIGHT = 'UP_RIGHT'
    RIGHT = 'RIGHT'
    DOWN_RIGHT = 'DOWN_RIGHT'
    DOWN = 'DOWN'
    DOWN_LEFT = 'DOWN_LEFT'
    LEFT = 'LEFT'
    UP_LEFT = 'UP_LEFT'

    @staticmethod
    def fromInt(value):
        if not isinstance(value, int):
            raise ValueError('Argument is not an int: ' + str(value))

        for i, dir in enumerate(Direction):
            if i == value:
                return dir

        raise ValueError('Argument is not in range of 0...7: ' + str(value))

