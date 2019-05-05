import numpy as np
import bs4
from bs4 import BeautifulSoup

from GameSettings import GameSettings
from GameState import GameState

class Parser():

    @staticmethod
    def parse(xml_string, lastGameState=None):
        if type(xml_string) is not str:
            return False
        
        soup = BeautifulSoup(xml_string, 'xml')
        
        settingsChanged = False
        gameStateResult = (False, lastGameState)
        moveRequestIssued = False
        
        
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

        return settingsChanged, gameStateResult, moveRequestIssued
                
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
        
        color = data.get('color')
        if color is not None:
            changed = color != GameSettings.ourColor
            GameSettings.ourColor = color
        
        return changed
    
    
    @staticmethod
    def parseState(data, lastGameState=None):
        if data is None:
            return None
        
        # parse into game settings
        
        startPlayerColor = data.state.get('startPlayerColor')
        if startPlayerColor is not None:
            GameSettings.startPlayerColor = startPlayerColor.lower()
        
        # parse into game state
        newGameState = GameState.copy(lastGameState) if lastGameState is not None else GameState()
        
        currentPlayerColor = data.state.get('currentPlayerColor')
        if currentPlayerColor is not None:
            newGameState.currentPlayerColor = currentPlayerColor.lower()
            
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
            gameState.board = np.zeros((10, 10), dtype=np.int)            

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
                
                # TODO: Check x and y for valid range
                if state == 'EMPTY':
                    gameState.board[x, y] = 0
                elif state == 'RED':
                    gameState.board[x, y] = 1 # TODO: this should always be the opponent
                elif state == 'BLUE':
                    gameState.board[x, y] = 2 # TODO: this should always be us
                elif state == 'OBSTRUCTED':
                    gameState.board[x, y] = 3
            else:
                print(repr(x), repr(y), repr(state))

        return True