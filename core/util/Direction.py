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

    def toAngle(self):
        if self.value == 'UP':
            return 0
        elif self.value == 'UP_RIGHT':
            return 45
        elif self.value == 'RIGHT':
            return 90
        elif self.value == 'DOWN_RIGHT':
            return 135
        elif self.value == 'DOWN':
            return 180
        elif self.value == 'DOWN_LEFT':
            return 225
        elif self.value == 'LEFT':
            return 270
        elif self.value == 'UP_LEFT':
            return 315

    def rotate(self, angle):
        assert angle % 45 == 0
        return Direction(Direction.fromAngle((self.toAngle() + angle) % 360))

    @staticmethod
    def fromInt(value):
        if not isinstance(value, int):
            raise ValueError('Argument is not an int: ' + str(value))

        for i, dir in enumerate(Direction):
            if i == value:
                return dir

        raise ValueError('Argument is not in range of 0...7: ' + str(value))

    @staticmethod
    def fromAngle(angle):
        if angle == 0:
            return Direction.UP
        elif angle == 45:
            return Direction.UP_RIGHT
        elif angle == 90:
            return Direction.RIGHT
        elif angle == 135:
            return Direction.DOWN_RIGHT
        elif angle == 180:
            return Direction.DOWN
        elif angle == 225:
            return Direction.DOWN_LEFT
        elif angle == 270:
            return Direction.LEFT
        elif angle == 315:
            return Direction.UP_LEFT

