class GameSettings():
    roomId = None
    ourColor = None
    startPlayerColor = None
    
    @staticmethod
    def reset():
        GameSettings.roomId = None
        GameSettings.ourColor = None
        GameSettings.startPlayerColor = None
        
    @staticmethod
    def __str__():
        stringRepresentation = 'GameSettings(\n'
        stringRepresentation += ' ' * 4 + 'roomId: ' + str(GameSettings.roomId) + '\n'
        stringRepresentation += ' ' * 4 + 'ourColor: ' + str(GameSettings.ourColor) + '\n'
        stringRepresentation += ' ' * 4 + 'startPlayerColor: ' + str(GameSettings.startPlayerColor) + '\n'
        stringRepresentation += ')'
        return stringRepresentation