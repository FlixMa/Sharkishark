#!/usr/local/bin/python3

import core
import time
import numpy as np
import random

class EvenSimplerGameLogicDelegate(core.logic.GameLogicDelegate):

    def __init__(self):
        self.currentGameState = None

    def onGameStateUpdate(self, gameState):
        super().onGameStateUpdate(gameState)
        self.currentGameState = gameState

    def onMoveRequest(self):
        super().onMoveRequest()
        if self.currentGameState is None:
            print('there is no field')
            return None
        else:
            print('issuing move')


            board = self.currentGameState.board

            myFieldState = core.util.FieldState.fromPlayerColor(core.state.GameSettings.ourColor)

            positions = np.argwhere(board == myFieldState)

            position = random.choice(positions)
            print(position)

            return core.util.Move(position[0], position[1], core.util.Direction.DOWN_LEFT)

gameLogic = EvenSimplerGameLogicDelegate()
gameClient = core.communication.GameClient('127.0.0.1', 13055, gameLogic)

gameClient.start()
gameClient.join()

while not gameClient.is_stopped():
    try:
        time.sleep(100)
    except:
        gameClient.stop()

#time.sleep(15)
#gameClient.stop()
