import numpy as np
from bs4 import BeautifulSoup

from ..state import *
from ..util import PlayerColor, FieldState

LOG_COUNT = 0

class Parser():

    @staticmethod
    def parse(xml_string, lastGameState=None, debug=False):
        global LOG_COUNT

        if type(xml_string) is not str:
            return False

        if debug:
            with open('./log/log_%d_raw.xml' % LOG_COUNT, 'w') as logfile:
                logfile.write(xml_string)

        soup = BeautifulSoup('<root>' + xml_string + '</root>', 'xml')

        if debug:
            with open('./log/log_%d_prettified.xml' % LOG_COUNT, 'w') as logfile:
                logfile.write(soup.prettify())
                LOG_COUNT += 1

        settingsChanged = False
        gameStateResult = (False, lastGameState)
        moveRequestIssued = False
        gameResult = None

        settingsChanged = Parser.parseRoomId(soup) or settingsChanged

        for roomElem in soup.find_all('room', roomId=GameSettings.roomId):

            cls = roomElem.data.get('class')

            #################################################
            if cls == 'welcomeMessage':
                # the server tells us which color we are
                settingsChanged = Parser.parseWelcomeMessage(roomElem.data) or settingsChanged


            #################################################
            elif cls == 'memento':
                # the server tells us the current game state
                newGameState = Parser.parseState(roomElem.data, lastGameState)
                if newGameState is not None:
                    gameStateResult = (True, newGameState)


            #################################################
            elif cls == 'sc.framework.plugins.protocol.MoveRequest':
                moveRequestIssued = True
            elif cls == 'result':
                gameResult = Parser.parseResult(roomElem.data)
            elif cls == 'error':
                if gameResult is None:
                    gameResult = Parser.parseError(roomElem.data)
            else:
                print('Parser found room element with unknown content:')
                print(roomElem.prettify())
                print()

        if debug:
            print('-'*50)
            print(soup.prettify())
            print('-'*50)

        return settingsChanged, gameStateResult, moveRequestIssued, gameResult

    @staticmethod
    def parseRoomId(soup):
        if soup is None:
            return False

        changed = False

        joinedTag = soup.find('joined')
        if joinedTag is not None:
            roomId = joinedTag.get('roomId')
            if roomId is not None:
                changed = roomId != GameSettings.roomId
                GameSettings.roomId = roomId

        return changed


    @staticmethod
    def parseWelcomeMessage(data):
        if data is None:
            return False

        changed = False

        color = PlayerColor.fromString(data.get('color'))
        if color is not None:
            changed = color != GameSettings.ourColor
            GameSettings.ourColor = color

        return changed


    @staticmethod
    def parseState(data, lastGameState=None):
        if data is None:
            return None

        # parse into game settings

        startPlayerColor = PlayerColor.fromString(data.state.get('startPlayerColor'))
        if startPlayerColor is not None:
            GameSettings.startPlayerColor = startPlayerColor

        # parse into game state
        newGameState = GameState.copy(lastGameState) if lastGameState is not None else GameState()

        currentPlayerColor = PlayerColor.fromString(data.state.get('currentPlayerColor'))
        if currentPlayerColor is not None:
            newGameState.currentPlayerColor = currentPlayerColor

        turn = data.state.get('turn')
        if turn is not None:
            try:
                turn = int(turn)
                newGameState.turn = turn
            except:
                pass

        Parser.parseBoard(data.board, newGameState)

        if lastGameState is None or newGameState != lastGameState:
            return newGameState
        else:
            return None


    @staticmethod
    def parseBoard(boardTag, gameState):
        if boardTag is None:
            return False

        if gameState.board is None:
            # TODO: remove hardcoded board size
            gameState.board = np.zeros((10, 10), dtype=FieldState)

        for field in boardTag.find_all('field'):
            x = field.get('x')
            y = field.get('y')
            state = field.get('state')

            if x is not None and y is not None:
                try:
                    x = int(x)
                    y = int(y)
                except Exception as ex:
                    print(ex)
                    continue

                gameState.board[x, y] = FieldState.fromString(state)
                '''
                # TODO: Check x and y for valid range
                if state == 'EMPTY':
                    gameState.board[x, y] = 0
                elif state == 'RED':
                    gameState.board[x, y] = 1 # TODO: this should always be the opponent
                elif state == 'BLUE':
                    gameState.board[x, y] = 2 # TODO: this should always be us
                elif state == 'OBSTRUCTED':
                    gameState.board[x, y] = 3
                '''
            else:
                print(repr(x), repr(y), repr(state))

        return True

    @staticmethod
    def parseResult(data):
        if data is None:
            return None

        winner = data.find('winner')
        winningPlayer = None
        if winner is not None:
            winningPlayer = PlayerColor.fromString(winner.get('color'))
            print('Winner:', winningPlayer)

        if winningPlayer is None:
            return (GameResult.TIED, GameResultCause.REGULAR, None)

        weWon = GameSettings.ourColor == winningPlayer

        ourScore = None
        theirScore = None

        for score in data.find_all('score'):
            cause = GameResultCause.fromString(score.get('cause'))
            reasonString = score.get('reason')
            playerResult = None
            playerPoints = None

            for i, part in enumerate(score.find_all('part')):
                content = None
                try:
                    content = int(part.string)
                except Exception as e:
                    print(e)
                if i == 0: # which player: 2 won, 1 tie, 0 lost
                    playerResult = content
                elif i == 1: # count
                    playerPoints = content

            playerResult = GameResult.fromInt(playerResult)

            parsedScore = (cause, reasonString, playerResult, playerPoints)
            if (weWon and playerResult == GameResult.WON) or (not weWon and GameResult.LOST):
                ourScore = parsedScore
            else:
                theirScore = parsedScore

        gameResultCause     = theirScore[0] if weWon else ourScore[0]
        gameResultReason    = theirScore[1] if weWon else ourScore[1]
        gameResult          = ourScore[2]

        if gameResult is None and gameResultCause is None and gameResultReason is None:
            return None

        return (gameResult, gameResultCause, gameResultReason)

    @staticmethod
    def parseError(data):
        return (GameResult.LOST, GameResultCause.RULE_VIOLATION, data.get('message'))





#
