from AsynchronousSocketClient import AsynchronousSocketClient
from GameLogicDelegate import GameLogicDelegate
from Parser import Parser

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
            self.gameLogicDelegate.onMoveRequest()
            
    def join(self, reservationCode=None):
        if reservationCode is None:
            self.send('<protocol><join gameType="swc_2019_piranhas"/>')
        else:
            raise NotImplementedError('Joining a game through reservation code is not yet supported. (RC: %s)' % reservationCode)
            
            
    def move(self, roomId, posX, posY, directionString):

        #let hints = move.debugHints.reduce(into: "") { $0 += "<hint content=\"\($1)\" />" }
        #let mv = "<data class=\"move\" x=\"\(move.x)\" y=\"\(move.y)\" direction=\"\(move.direction)\">\(hints)</data>"
        #self.socket.send(message: "<room roomId=\"\(self.roomId!)\">\(mv)</room>")
        hintsXML = ''
        self.send('<room roomId="%s"><data class="move" x="%d" y="%d" direction="%s">%s</data></room>' % (roomId, posX, posY, directionString, hintsXML))
        