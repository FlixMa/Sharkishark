import gym
import numpy as np
from math import sqrt
from gym import error, spaces, utils
from gym.utils import seeding
from core.logic import GameLogicDelegate
from core.util import FieldState, PlayerColor, Direction, Move
from core.state import GameSettings
from threading import Event



class PiranhasEnv(gym.Env, GameLogicDelegate):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # members from gym.Env
        self.name = 'piranhas'

        self.action_space = spaces.Tuple(
            (spaces.Discrete(10), spaces.Discrete(10), spaces.Discrete(8)))

        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0, high=1, shape=(10, 10, 3), dtype=np.uint8
            ),
            "opponent": spaces.Dict({
                "nr_fish": spaces.Discrete(16),
                "biggest_group": spaces.Discrete(16)
            }),
            "me": spaces.Dict({
                "nr_fish": spaces.Discrete(16),
                "biggest_group": spaces.Discrete(16)
            }),
            "turn_nr": spaces.Discrete(30),
            "begins": spaces.Discrete(1)
        }),
        self.observation = {
            "board": np.zeros((10, 10, 3)),
            "opponent": {
                "nr_fish": 16,
                "biggest_group": 8,
            },
            "me": {
                "nr_fish": 16,
                "biggest_group": 8
            },
            "turn_nr": 0,
            "begins": 0
        }

        # members from GameLogicDelegate
        super(GameLogicDelegate).__init__()
        self.game_state_update_event = Event()
        self.move_request_event = Event()
        self.move_decision_taken_event = Event()
        self.global_move = None  # of type Move()

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
             action (spaces.Tuple): an action provided by the agent
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

        # 1. aktuellen game state auslesen (nur beim ersten mal hier oben)

        # in the first round only (not None)

        # wait for move request
        self.move_request_event.wait()
        self.move_request_event.clear()

        # send move request (somehow)
        if GameSettings.ourColor != GameSettings.startPlayerColor:
            self.global_move = Move(action[0], action[1], Direction(action[2]))
        else:
            self.global_move = Move(action[0], action[1], Direction(action[2]))
        # self.global_move = self.move()  # this is nn.forward(observation) --> TODO richtig so?
        previous_game_state = self.currentGameState  # remember the last game state
        self.move_decision_taken_event.set()  # onMoveRequest listens for this event

        # wait until game state has been reported
        # what the opponent did -> calc reward based on that too
        self.game_state_update_event.wait()
        self.game_state_update_event.clear()

        # calculate reward
        reward, legal_move = self.calc_reward(previous_game_state)

        return (self.observation, reward, not legal_move, {})  # TODO check if game ended

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        # TODO restart game?
        self.game_state_update_event.wait()
        self.game_state_update_event.clear()
        self.observation["begins"] = GameSettings.startPlayerColor
        # TODO flag für Spiel zuende

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

    # Overridden methods inherited from GameLogicDelegate

    def onGameStateUpdate(self, game_state):
        super().onGameStateUpdate(game_state)
        self.currentGameState = game_state

        # preprocessing
        self.observation["board"] = np.zeros((10, 10, 3))  # (us, opponent, kraken)
        print(self.observation["board"].shape, self.currentGameState.board.shape)
        self.observation["board"][:, :, 0] = self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor)
        self.observation["board"][:, :, 1] = self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor)
        self.observation["board"][:, :, 2] = self.currentGameState.board == FieldState.OBSTRUCTED

        print(self.observation["board"].shape, self.currentGameState.board.shape)

        if GameSettings.ourColor != GameSettings.startPlayerColor:
            # we normalize the board so that we are always the starting player
            # who has fishes on the left and right hand side
            self.observation["board"] = np.rot90(self.observation["board"])
        self.observation["board"].astype('uint8')  # saves storage in experience memory

        # TODO? opponent["nr_fish"] and opponent["biggest_group"]
        self.observation["turn_nr"] = self.currentGameState.turn

        self.game_state_update_event.set()

    def onMoveRequest(self):
        super().onMoveRequest()
        if self.currentGameState is None:
            print('there is no field')
            return None
        else:
            self.move_request_event.set()
            # wait until there is a move decision
            self.move_decision_taken_event.wait()
            self.move_decision_taken_event.clear()
            print('issuing move')

            return self.global_move

    # Helper methods
    @staticmethod
    def norm(a):
        return sqrt(a[0] ** 2 + a[1] ** 2)

    @staticmethod
    def calc_mean_distance(fishes):
        mean_coordinate = np.array([0, 0])
        for fish_coord in fishes:
            mean_coordinate += fish_coord

        mean_coordinate /= len(fishes)

        mean_distance = 0
        for fish in fishes:
            mean_distance += PiranhasEnv.norm(mean_coordinate - fish)
        return mean_distance / len(fishes)

    @staticmethod
    def calc_mean_distance_using_median_center(fishes):
        median_coordinate = np.median(fishes, axis=0)

        squared_error = 0
        for fish_coord in fishes:
            squared_error += np.square(median_coordinate - fish_coord).sum()
        return squared_error / len(fishes)

    @staticmethod
    def get_biggest_group(fishes):
        considered_fishes = []
        unconsidered_fishes = fishes

        # for fish in fishes:

    @staticmethod
    def get_eaten_fish_reward(own_fishes_previous, own_fishes_current,
                              opp_fishes_previous, opp_fishes_current):
        return len(own_fishes_current) - len(own_fishes_previous) + \
               len(opp_fishes_previous) - len(opp_fishes_current)

    def calc_reward(self, previous_game_state):
        # compare current board to last board
        own_fishes_previous = np.argwhere(
            previous_game_state.board == FieldState.fromPlayerColor(GameSettings.ourColor))
        own_fishes_current = np.argwhere(
            self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor))

        # opponent's fishes
        opp_fishes_previous = np.argwhere(
            previous_game_state.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor))
        opp_fishes_current = np.argwhere(
            self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor))

        mean_distance_previous = PiranhasEnv.calc_mean_distance_using_median_center(own_fishes_previous)
        mean_distance_current = PiranhasEnv.calc_mean_distance_using_median_center(own_fishes_current)
        #TODO: use calc_mean_distance_using_median_center?

        # biggest_group_previous = PiranhasEnv.get_biggest_group(own_fishes_previous)
        # biggest_group_current = PiranhasEnv.get_biggest_group(own_fishes_current)

        reward_fish_eaten = PiranhasEnv.get_eaten_fish_reward(
            own_fishes_previous, own_fishes_current,
            opp_fishes_previous, opp_fishes_current)

        valid_move = self.validateMove(self.global_move)
        valid_move_reward = 0
        if not valid_move:
            valid_move_reward = -1000

        return (mean_distance_previous - mean_distance_current) + \
               reward_fish_eaten + valid_move_reward, valid_move
               # (biggest_group_current - biggest_group_previous) + \
