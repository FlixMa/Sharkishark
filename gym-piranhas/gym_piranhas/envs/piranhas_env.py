import gym
import numpy as np
import subprocess
from math import sqrt
from gym import error, spaces, utils
from gym.utils import seeding
from core.communication import GameClient
from core.logic import GameLogicDelegate
from core.util import FieldState, PlayerColor, Direction, Move
from core.state import GameSettings, GameResult, GameResultCause, GameState
from threading import Event


class PiranhasEnv(gym.Env, GameLogicDelegate):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # members from gym.Env
        self.name = 'piranhas'
        self.debug = False
        # full steam ahead
        self.action_space = spaces.Discrete(800)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1000,),
            dtype=np.float
        )
        self.observation = None
        self.num_consecutive_invalid_moves = 0
        self.num_consecutive_valid_moves = 0

        # members from GameLogicDelegate
        super(GameLogicDelegate).__init__()
        self.game_state_update_event = Event()
        self.move_request_event = Event()
        self.move_decision_taken_event = Event()
        self.global_move = None  # of type Move()
        self.reset_callback = None
        self.result = None
        self.cause = None

        self.simulation_is_active = False

        self.game_client = None
        self.host = '127.0.0.1'
        self.port = 13050
        self.opponents_executable = "java -jar ../simpleclient-piranhas-19.2.1.jar --host {host} --port {port}"

        # numpy random object
        self.np_random = None

    def set_reset_callback(self, reset_callback):
        self.reset_callback = reset_callback

    def set_debug(self, debug):
        self.debug = True if debug else False

    # Overridden methods inherited from gym.Env
    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
                number generators. The first value in the list should be the
                "main" seed, or the value which a reproducer should pass to
                'seed'. Often, the main seed equals the provided 'seed', but
                this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
             action (int): an action provided by the agent
         Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        '''
        Abfolge:

            1. aktuellen game state auslesen
                1.1. game state normalisieren (z.b. spielfeld drehen wenn wir nicht blau sind)
            2. prediction machen (unser nn)
                2.1. action post processing (z.b. aktion 'wieder entdrehen')
            3. zug validieren (sonst böse bestrafung)
            4. zug an den server senden
            5. auf gegner (neues board) warten
            6. reward berechnen

            -> bei 1 weitermachen
        '''

        if self.simulation_is_active:
            return self.simulate_step(action)

        # 1. aktuellen game state auslesen (nur beim ersten mal hier oben)

        # in the first round only (not None)

        # wait for move request
        if self.debug:
            print("[env] Waiting for move request ... ")
        self.move_request_event.wait()
        if self.debug:
            print("[env] Received move request. ")
        self.move_request_event.clear()

        # send move request (somehow)
        if self.debug:
            print("[env] Calculating move ... ")
        if GameSettings.ourColor == GameSettings.startPlayerColor:
            self.global_move = PiranhasEnv.retrieve_move(action)
        else:
            self.global_move = PiranhasEnv.retrieve_move(action, rotate=True)

        # remember the last game state for reward
        previous_game_state = self.currentGameState
        self.move_decision_taken_event.set()  # onMoveRequest listens for this event
        if self.debug:
            print("[env] Move decision set. ")

        # wait until game state has been reported
        # what the opponent did -> calc reward based on that too
        if self.debug:
            print("[env] Waiting for game state update ... ")
        self.game_state_update_event.wait()
        if self.debug:
            print("[env] Received game state update. ")
        self.game_state_update_event.clear()

        # calculate reward
        if self.debug:
            print("[env] Calculating reward ... ")
        reward, done = self.calc_reward(previous_game_state)
        print("\n[env] Reward: {:.2f}; GameResult: {}; Cause: {}".format(
            reward, self.result, self.cause))

        return self.observation, reward, done, {}

    def simulate_step(self, action):
        if not self.simulation_is_active:
            raise IllegalStateException('Can\'t simulate a step when not being in simulation mode.')

        if GameSettings.ourColor == GameSettings.startPlayerColor:
            self.global_move = PiranhasEnv.retrieve_move(action)
        else:
            self.global_move = PiranhasEnv.retrieve_move(action, rotate=True)

        previous_game_state = self.currentGameState

        next_game_state = self.currentGameState.apply(self.global_move)

        user_info = {'simulation_is_active': self.simulation_is_active}

        if next_game_state is None:
            return self.observation, -100.0, True, user_info

        self.currentGameState = next_game_state
        reward, done = self.calc_reward(previous_game_state)
        self.observation = PiranhasEnv.convert_game_state(next_game_state)
        return self.observation, reward, done, user_info

    def reset(self, initial_game_state=None):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        if self.debug:
            print("[env] Resetting environment ... ")
        self.currentGameState = None
        self.result = None
        self.cause = None
        self.num_consecutive_valid_moves = 0
        self.num_consecutive_invalid_moves = 0

        self.game_state_update_event.clear()
        self.move_request_event.clear()
        self.move_decision_taken_event.clear()


        if self.simulation_is_active:
            if not isinstance(initial_game_state, GameState):
                raise ValueError('No initial GameState given. Got: ' + str(type(initial_game_state)))
            self.currentGameState = initial_game_state
            self.observation = PiranhasEnv.convert_game_state(initial_game_state)
        else:
            self.makeNewGame()

            if self.debug:
                print("[env] Waiting for initial game state ... ")
            self.game_state_update_event.wait()
            self.game_state_update_event.clear()
            if self.debug:
                print("[env] Received initial game state. ")

        return self.observation

    def render(self, mode='human', close=False):
        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        pass

    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def makeNewGame(self):
        if self.simulation_is_active:
            return

        self.game_client = GameClient(self.host, self.port, self)
        self.game_client.start()
        self.game_client.join()
        subprocess.Popen(
            self.opponents_executable.format(host=self.host, port=self.port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True
         )

    # Overridden methods inherited from GameLogicDelegate

    def onGameStateUpdate(self, game_state):
        super().onGameStateUpdate(game_state)
        self.currentGameState = GameState.copy(game_state)

        # preprocessing
        if self.debug:
            print("[env] Preprocessing ... ")
        self.observation = PiranhasEnv.convert_game_state(game_state)

        self.game_state_update_event.set()
        if self.debug:
            print("[env] Preprocessing done.")

    def onMoveRequest(self):
        super().onMoveRequest()
        if self.currentGameState is None:
            print('[env] there is no field')
            return None
        else:
            self.move_request_event.set()
            # wait until there is a move decision
            if self.debug:
                print("[env] Waiting for a move decision ... ")
            self.move_decision_taken_event.wait()
            self.move_decision_taken_event.clear()
            if self.debug:
                print('[env] issuing move {}.'.format(self.global_move))

            return self.global_move

    def onGameResult(self, result, cause, description):
        if self.debug:
            print("[env] Received gameResult '({}, {})'".format(result, cause))
        self.result = result
        self.cause = cause
        # hide this as a game state
        # # set ALL the events
        self.game_state_update_event.set()
        self.move_request_event.set()
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


    @staticmethod
    def retrieve_move(action, rotate=False):
        try:
            action = int(action)
        except Exception as ex:
            raise ValueError('Got action of invalid type. Expected int, got: ' + str(type(action)) + ' -> ' + str(action) + '\n' + str(ex))

        direction = Direction.fromInt(int(action % 8))
        move_x = (action // 8) // 10
        move_y = (action // 8) % 10

        if rotate:
            # rotate clockwise; (0,0) is bottom-left
            tmp = np.zeros((10, 10))
            tmp[9 - move_y, move_x] = 1
            tmp_rot = np.rot90(tmp, 3)  # numpy is counter-clockwise per default
            coords = np.argwhere(tmp_rot)
            move_x = int(coords[0][1])
            move_y = 9 - int(coords[0][0])
            direction = direction.rotate(90)

        return Move(move_x, move_y, direction)

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
