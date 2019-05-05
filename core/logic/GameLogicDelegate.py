from ..state import GameSettings

class GameLogicDelegate():

    def onSettingsUpdate(self):
        print('-> onSettingsUpdate()')
        print(GameSettings())

    def onGameStateUpdate(self, gameState):
        print('-> onGameStateUpdate()')
        print(gameState)

    def onMoveRequest(self):
        print('-> onMoveRequest()')
