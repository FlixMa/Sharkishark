from ..state import GameSettings

class GameLogicDelegate():

    def onSettingsUpdate(self):
        print('-> onSettingsUpdate()')
        print(GameSettings())

    def onGameStateUpdate(self, gameState):
        print('-> onGameStateUpdate()')
        gameState.printColored()

    def onMoveRequest(self):
        print('-> onMoveRequest()')