import numpy as np
import subprocess
import operator
from time import time
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

    def make_new_game(self, start_opponent=False):
        self.game_client = GameClient(self.host, self.port, self)
        self.game_client.start()
        self.game_client.join(self.reservation)

        if start_opponent or self.autoplay:
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
        #super().onGameStateUpdate(game_state)
        self.currentGameState = GameState.copy(game_state)

    def onMoveRequest(self):
        #super().onMoveRequest()
        start_time = time()

        if self.currentGameState is None:
            print('[env] there is no field')
            return None

        our_fishes = self.currentGameState.get_fishes(GameSettings.ourColor)
        estimated_rewards = {}  # dict of (move, reward)
        for fish in our_fishes:
            for dir in Direction:
                move = Move(fish[0], fish[1], dir)
                game_state = self.currentGameState.apply(move)
                if game_state is None:
                    # this move was invalid
                    continue
                their_fishes = game_state.get_fishes(GameSettings.ourColor.otherColor())

                possible_rewards = []
                for their_fish in their_fishes:
                    for their_dir in Direction:
                        their_move = Move(their_fish[0], their_fish[1], their_dir)
                        next_game_state = game_state.apply(their_move)
                        if next_game_state is None:
                            # this move was invalid
                            return
                        reward, done, _ = self.currentGameState.estimate_reward(next_game_state)
                        possible_rewards.append(reward)

                estimated_rewards[move] = np.array(possible_rewards)

        worst_case_move = None
        highest_worst_case_reward = None

        best_case_move = None
        highest_best_case_reward = None

        typical_move = None
        highest_typical_reward = None
        for move, possible_rewards in estimated_rewards.items():
            typical_reward = possible_rewards.mean()
            best_case_reward = possible_rewards.max()
            worst_case_reward = possible_rewards.min()

            if highest_typical_reward is None or typical_reward > highest_typical_reward:
                typical_move = move
                highest_typical_reward = typical_reward

            if highest_best_case_reward is None or best_case_reward > highest_best_case_reward:
                best_case_move = move
                highest_best_case_reward = best_case_reward

            if highest_worst_case_reward is None or worst_case_reward > highest_worst_case_reward:
                worst_case_move = move
                highest_worst_case_reward = worst_case_reward

        print(
        '''[env] Sending move after {:.3f} seconds. Expected Reward:
            Typical:    {:10.2f} {}
            Best Case:  {:10.2f} {}
            Worst Case: {:10.2f} {}
        '''.format(
            time()-start_time,
            highest_typical_reward, typical_move,
            highest_best_case_reward, best_case_move,
            highest_worst_case_reward, worst_case_move
        ))
        return typical_move

    def onGameResult(self, result, cause, description):
        print("[env] Received gameResult '({}, {})'".format(result, cause))
        self.result = result
        self.cause = cause

        if self.autoplay:
            self.make_new_game()
        return True

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
