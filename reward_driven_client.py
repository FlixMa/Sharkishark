import numpy as np
import subprocess
import operator
from core.communication import GameClient
from core.logic import GameLogicDelegate
from core.util import FieldState, PlayerColor, Direction, Move
from core.state import GameSettings, GameResult, GameResultCause, GameState


class RewardDrivenClient(GameLogicDelegate):

    def __init__(self, host, port, reservation, autoplay=False):
        # members from GameLogicDelegate
        super(GameLogicDelegate).__init__()
        self.reset_callback = None
        self.result = None
        self.cause = None
        self.autoplay = autoplay

        self.game_client = None
        self.host = host
        self.port = port
        self.reservation = reservation
        self.opponents_executable = "java -jar ../piranhas-not-so-simple-client-19.2.1.jar --host {host} --port {port}"

        # numpy random object
        self.np_random = None

    def make_new_game(self):
        self.game_client = GameClient(self.host, self.port, self)
        self.game_client.start()
        self.game_client.join(self.reservation)

        if self.autoplay:
            # start opponent
            subprocess.Popen(
                self.opponents_executable.format(host=self.host, port=self.port),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
             )
        self.game_client.wait_until_stopped()

    # Overridden methods inherited from GameLogicDelegate

    def onGameStateUpdate(self, game_state):
        super().onGameStateUpdate(game_state)
        self.currentGameState = GameState.copy(game_state)

    def onMoveRequest(self):
        super().onMoveRequest()
        if self.currentGameState is None:
            print('[env] there is no field')
            return None
        else:
            our_fishes = np.argwhere(self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor))
            estimated_rewards = {}  # dict of (move, reward)
            for fish in our_fishes:
                for dir in Direction:
                    move = Move(fish[0], fish[1], dir)
                    estimated_rewards[move] = self.currentGameState.estimate_reward(move)

            best_move = max(estimated_rewards.items(), key=operator.itemgetter(1))
            return best_move[0]

    def onGameResult(self, result, cause, description):
        print("[env] Received gameResult '({}, {})'".format(result, cause))
        self.result = result
        self.cause = cause

        if self.autoplay:
            self.make_new_game()
        return True

    # Helper methods
    @staticmethod
    def convert_game_state(game_state):
        if GameSettings.ourColor != GameSettings.startPlayerColor:
            # we normalize the board so that we are always the starting player
            # who has fishes on the left and right hand side
            game_state.board = np.rot90(game_state.board)

        observation = np.zeros((10 * 10 * (8 + 2)), dtype=np.bool)
        positions = np.argwhere(game_state.board == FieldState.fromPlayerColor(GameSettings.ourColor))

        dir_enumerated = list(enumerate(Direction))
        for x, y in positions:
            for i, dir in dir_enumerated:
                move = Move(x, y, dir)
                idx = (x + (y * 10)) * 8 + i
                observation[idx] = move.validate(game_state)[0]

        observation[-200:-100] = (game_state.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor())).flatten()
        observation[-100:] = (game_state.board == FieldState.OBSTRUCTED).flatten()

        return observation

    def calc_reward(self, previous_game_state):
        if self.result == GameResult.WON and \
                self.cause == GameResultCause.REGULAR:
            return 100.0, True
        elif self.result == GameResult.LOST and \
                (self.cause == GameResultCause.REGULAR or
                 self.cause == GameResultCause.RULE_VIOLATION):
            return -100.0, True

        estimated_reward, done = previous_game_state.estimate_reward(self.currentGameState)

        return estimated_reward, done or self.result is not None


if __name__ == "__main__":
    client = RewardDrivenClient('127.0.0.1', 13050, None, True)

    client.make_new_game()