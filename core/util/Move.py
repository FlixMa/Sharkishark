class Move():

    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    def __str__(self):
        return "Move(({}, {}) in direction {})".format(
            self.x, self.y, self.direction)