from .AsynchronousSocketClient import AsynchronousSocketClient
from ..logic import GameLogicDelegate
from ..parsing import Parser
from ..state import GameSettings

class GameClient(AsynchronousSocketClient):

    def __init__(self, host, port, gameLogicDelegate):
        super().__init__(host, port)
        self.gameLogicDelegate = gameLogicDelegate

    def onMessage(self, message):
        if not isinstance(self.gameLogicDelegate, GameLogicDelegate):
            return

        settingsChanged, (gameStateChanged, state), moveRequestIssued = Parser.parse(message)

        if settingsChanged:
            self.gameLogicDelegate.onSettingsUpdate()

        if gameStateChanged:
            self.gameLogicDelegate.onGameStateUpdate(state)

        if moveRequestIssued:
            result = self.gameLogicDelegate.onMoveRequest()
            if result is not None:
                self.move(GameSettings.roomId, result.x, result.y, result.direction.value)

    def join(self, reservationCode=None):
        # The join method is intentionally overidden. We don't want anybody to intercept the receiving thread.

        if reservationCode is None:
            self.send('<protocol><join gameType="swc_2019_piranhas"/>')
        else:
            self.send('<protocol><joinPrepared reservationCode="%s"/>' % reservationCode)


    def move(self, roomId, posX, posY, directionString):

        #let hints = move.debugHints.reduce(into: "") { $0 += "<hint content=\"\($1)\" />" }
        #let mv = "<data class=\"move\" x=\"\(move.x)\" y=\"\(move.y)\" direction=\"\(move.direction)\">\(hints)</data>"
        #self.socket.send(message: "<room roomId=\"\(self.roomId!)\">\(mv)</room>")
        hintsXML = '<hint content="noch ein Hint" />'
        message = '<room roomId="%s"><data class="move" x="%d" y="%d" direction="%s">%s</data></room>' % (roomId, posX, posY, directionString, hintsXML)
        print('\nsending:', message, '\n')

        self.send(message)
