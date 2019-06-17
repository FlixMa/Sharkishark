import gym
import numpy as np
from math import sqrt
from gym import error, spaces, utils
from gym.utils import seeding
from core.logic import GameLogicDelegate
from core.util import FieldState, PlayerColor, Direction, Move
from core.state import GameSettings, GameResult, GameResultCause, GameState
from threading import Event


class PiranhasEnv(gym.Env, GameLogicDelegate):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # members from gym.Env
        self.name = 'piranhas'

        # full steam ahead
        self.action_space = spaces.Discrete(800)

        self.observation_space = spaces.Box(
                low=0, high=1, shape=(10, 10, 3), dtype=np.uint8
            ),
        self.observation = np.zeros((10, 10, 3))

        # members from GameLogicDelegate
        super(GameLogicDelegate).__init__()
        self.game_state_update_event = Event()
        self.move_request_event = Event()
        self.move_decision_taken_event = Event()
        self.global_move = None  # of type Move()
        self.reset_callback = None
        self.result = None
        self.cause = None

        # numpy random object
        self.np_random = None

    def set_reset_callback(self, reset_callback):
        self.reset_callback = reset_callback

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

        # 1. aktuellen game state auslesen (nur beim ersten mal hier oben)

        # in the first round only (not None)

        # wait for move request
        print("[env] Waiting for move request ... ")
        self.move_request_event.wait()
        print("[env] Received move request. ")
        self.move_request_event.clear()

        # send move request (somehow)
        print("[env] Calculating move ... ")
        if GameSettings.ourColor == GameSettings.startPlayerColor:
            self.global_move = PiranhasEnv.retrieve_move(action)
        else:
            self.global_move = PiranhasEnv.retrieve_move(action, rotate=True)

        # check if move is valid
        # -> let the net redo the step action if not
        # -> otherwise continue

        if self.currentGameState is not None:
            print("[env] Validating move locally... ")
            valid, destination = PiranhasEnv.validate_move(self.global_move, self.currentGameState)
            if not valid:
                print("[env] Move is invalid. ")
                self.move_request_event.set() # to not get stuck above
                return self.observation, -10., self.result is not None, {'locally_validated': False}

        # remember the last game state for reward
        previous_game_state = self.currentGameState
        self.move_decision_taken_event.set()  # onMoveRequest listens for this event
        print("[env] Move decision set. ")

        # wait until game state has been reported
        # what the opponent did -> calc reward based on that too
        print("[env] Waiting for game state update ... ")
        self.game_state_update_event.wait()
        print("[env] Received game state update. ")
        self.game_state_update_event.clear()

        # calculate reward
        print("[env] Calculating reward ... ")
        reward, done = self.calc_reward(previous_game_state)
        print("[env] Reward: {}; Done: {}".format(reward, done))

        return self.observation, reward, done, {'locally_validated': True}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        print("[env] Resetting environment ... ")
        self.currentGameState = None
        self.result = None
        self.cause = None
        self.reset_callback()
        self.game_state_update_event.clear()
        self.move_request_event.clear()
        self.move_decision_taken_event.clear()
        print("[env] Waiting for initial game state ... ")
        self.game_state_update_event.wait()
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

    # Overridden methods inherited from GameLogicDelegate

    def onGameStateUpdate(self, game_state):
        super().onGameStateUpdate(game_state)
        self.currentGameState = game_state

        # preprocessing
        print("[env] Preprocessing ... ")
        self.observation = np.zeros((10, 10, 3))  # (us, opponent, kraken)

        self.observation[:, :, 0] = self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor)
        self.observation[:, :, 1] = self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor)
        self.observation[:, :, 2] = self.currentGameState.board == FieldState.OBSTRUCTED

        if GameSettings.ourColor != GameSettings.startPlayerColor:
            # we normalize the board so that we are always the starting player
            # who has fishes on the left and right hand side
            self.observation = np.rot90(self.observation)
        self.observation.astype('uint8')  # saves storage in experience memory

        self.game_state_update_event.set()
        print("[env] Preprocessing done.")

    def onMoveRequest(self):
        super().onMoveRequest()
        if self.currentGameState is None:
            print('[env] there is no field')
            return None
        else:
            self.move_request_event.set()
            # wait until there is a move decision
            print("[env] Waiting for a move decision ... ")
            self.move_decision_taken_event.wait()
            self.move_decision_taken_event.clear()
            print('[env] issuing move {}.'.format(self.global_move))

            return self.global_move

    def onGameResult(self, result, cause, description):
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
    def retrieve_move(action, rotate=False):
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

    @staticmethod
    def validate_move(move, gameState=None):
        '''
            Returns a tuple:
            1. flag, if move is valid
            2. destination (x, y) of that move or None if invalid
        '''

        if not isinstance(move, Move):
            raise ValueError('move argument is not of type "core.util.Move". Given: ' + str(type(move)))

        if gameState is None:
            raise ValueError('gameState argument is None')
        elif not isinstance(gameState, GameState):
            raise ValueError('gameState argument is not of type "core.state.GameState". Given: ' + str(type(gameState)))

        if gameState is None:
            raise ValueError('No gameState found.')
        elif gameState.board is None:
            raise ValueError('No board found.')

        board = gameState.board

        '''
            check if this move is valid

            1. a fish of ours is selected
            2. the fish can move in that direction (not directly next to the bounds)
            3. destination is empty or opponent's fish (not ours and not a kraken)
            4. our fish does not jump over a opponent's fish
        '''

        ourFishFieldState = FieldState.fromPlayerColor(GameSettings.ourColor)

        if not (board[move.x, move.y] == ourFishFieldState):
            print('Can\'t mind control the opponents fishes :(')
            return False, None

        # count fishes in that row
        #
        # on which axis are we and
        # on that axis - where are we exactly?
        axis = None
        current_position_on_axis = None
        if move.direction == Direction.UP or move.direction == Direction.DOWN:
            axis = board[move.x]
            current_position_on_axis = move.y
        elif move.direction == Direction.LEFT or move.direction == Direction.RIGHT:
            axis = board[:, move.y]
            current_position_on_axis = move.x
        elif move.direction == Direction.DOWN_LEFT or move.direction == Direction.UP_RIGHT:
            axis = board.diagonal(move.y - move.x)
            current_position_on_axis = move.x if move.y > move.x else move.y
        elif move.direction == Direction.UP_LEFT or move.direction == Direction.DOWN_RIGHT:
            flippedX = ((board.shape[0] - 1) - move.x)

            # NOTE: flipud actually flips the board left to right because of the way how we index it
            axis = np.flipud(board).diagonal(move.y - flippedX)

            current_position_on_axis = flippedX if move.y > flippedX else move.y

        print('move', move.direction.name, (move.x, move.y), '-> axis: [ ', end='')
        for item in axis:
            print(item.name, end=' ')
        print('], idx:', current_position_on_axis)

        num_fishes = ((axis == FieldState.RED) | (axis == FieldState.BLUE)).sum()
        print('-> fishlis:', num_fishes)

        #  where do we wanna go?
        #  NOTE: y is upside down / inverted
        direction_forward = (move.direction in [Direction.UP, Direction.UP_LEFT, Direction.UP_RIGHT, Direction.RIGHT])
        destination_position_on_axis = (current_position_on_axis + num_fishes) if direction_forward else (current_position_on_axis - num_fishes)
        print('direction_forward:', direction_forward)
        print('destination:', destination_position_on_axis)

        # check for bounds
        if destination_position_on_axis < 0 or destination_position_on_axis >= axis.size:
            print('Exceeding bounds. %d of %d' % (destination_position_on_axis, axis.size))
            return False, None

        # what type is that destination field?
        destinationFieldState = axis[destination_position_on_axis]
        if destinationFieldState == FieldState.OBSTRUCTED or destinationFieldState == ourFishFieldState:
            print('Destination is obstructed or own fish:', destinationFieldState)
            return False, None

        # is an opponents fish in between(! excluding the destiantion !)?
        opponentsFieldState = FieldState.RED if ourFishFieldState == FieldState.BLUE else FieldState.BLUE
        for idx in range(current_position_on_axis, destination_position_on_axis, 1 if direction_forward else -1):
            if axis[idx] == opponentsFieldState:
                print('Can\'t jump over opponents fish.')
                return False, None


        dest_x, dest_y = move.x, move.y
        if move.direction == Direction.UP or move.direction == Direction.DOWN:
            dest_y = destination_position_on_axis
        elif move.direction == Direction.LEFT or move.direction == Direction.RIGHT:
            dest_x = destination_position_on_axis
        elif move.direction == Direction.DOWN_LEFT or move.direction == Direction.UP_RIGHT:
            dest_x += num_fishes * (1 if direction_forward else -1)
            dest_y += num_fishes * (1 if direction_forward else -1)
        elif move.direction == Direction.UP_LEFT or move.direction == Direction.DOWN_RIGHT:
            dest_x -= num_fishes * (1 if direction_forward else -1)
            dest_y += num_fishes * (1 if direction_forward else -1)

        return True, (dest_x, dest_y)

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
        median_coordinate = np.percentile(fishes,
                                          50,
                                          axis=0,
                                          interpolation='nearest')

        squared_error = 0
        for fish_coord in fishes:
            squared_error += np.square(median_coordinate - fish_coord).sum()
        return squared_error / len(fishes)

    @staticmethod
    def neighbors(x, y, boolBoard):

        xmin = max(0, x - 1)
        xmax = min(x + 2, boolBoard.shape[0])
        ymin = max(0, y - 1)
        ymax = min(y + 2, boolBoard.shape[1])

        neighborhood = boolBoard[xmin:xmax, ymin:ymax]
        # print('umfeld')
        # print(neighborhood)

        neighbors = np.argwhere(neighborhood)

        neighbors += np.array([xmin, ymin])

        return neighbors

    @staticmethod
    def find_groups(playerColor, board):
        boolBoard = (board == FieldState.fromPlayerColor(playerColor))
        fishesToConsider = np.argwhere(boolBoard)
        groups = []

        # calculate group for each starting position
        while len(fishesToConsider) > 0:
            fish = fishesToConsider[0]

            # get neighborhood of chosen fish and iteratively do a flood-fill
            fish_neighborhood = PiranhasEnv.neighbors(fish[0], fish[1], boolBoard)
            i = 0
            while i < len(fish_neighborhood):
                neighbor = fish_neighborhood[i]

                its_neighborhood = PiranhasEnv.neighbors(neighbor[0], neighbor[1], boolBoard)
                concatenated = np.concatenate((fish_neighborhood, its_neighborhood))
                fish_neighborhood = np.unique(concatenated, axis=0)
                i += 1

            groups.append(fish_neighborhood)

            # remove all fishes in this group from the fishes to consider next round
            remainingMask = ~(np.isin(list(map(lambda a: a[0] * 10 + a[1], fishesToConsider)),
                                      list(map(lambda a: a[0] * 10 + a[1], fish_neighborhood))))
            fishesToConsider = fishesToConsider[remainingMask]

        return groups

    @staticmethod
    def get_biggest_group(playerColor, board):
        groups = PiranhasEnv.find_groups(playerColor, board)
        largestGroup = None
        largestSize = 0
        for group in groups:
            size = len(group)
            if largestSize < size:
                largestSize = size
                largestGroup = group

        return largestGroup

    @staticmethod
    def get_eaten_fish_reward(own_fishes_previous, own_fishes_current,
                              opp_fishes_previous, opp_fishes_current):
        return len(own_fishes_current) - len(own_fishes_previous) + \
               len(opp_fishes_previous) - len(opp_fishes_current)

    def calc_reward(self, previous_game_state):
        if self.result == GameResult.WON and \
                self.cause == GameResultCause.REGULAR:
            return 10, True
        elif self.result == GameResult.LOST and \
                (self.cause == GameResultCause.REGULAR or
                 self.cause == GameResultCause.RULE_VIOLATION):
            return -10, True

        # compare current board to last board
        own_fishes_previous = np.argwhere(
            previous_game_state.board == FieldState.fromPlayerColor(GameSettings.ourColor))
        own_fishes_current = np.argwhere(
            self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor))

        # opponent's fishes
        opp_fishes_previous = np.argwhere(
            previous_game_state.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor()))
        opp_fishes_current = np.argwhere(
            self.currentGameState.board == FieldState.fromPlayerColor(GameSettings.ourColor.otherColor()))

        mean_distance_previous = PiranhasEnv.calc_mean_distance_using_median_center(own_fishes_previous)
        mean_distance_current = PiranhasEnv.calc_mean_distance_using_median_center(own_fishes_current)

        biggest_group_previous = PiranhasEnv.get_biggest_group(
            GameSettings.ourColor, self.currentGameState.board)
        biggest_group_current = PiranhasEnv.get_biggest_group(
            GameSettings.ourColor.otherColor(), self.currentGameState.board)

        # remote eaten fish
        reward_fish_eaten = PiranhasEnv.get_eaten_fish_reward(
            own_fishes_previous, own_fishes_current,
            opp_fishes_previous, opp_fishes_current)

        return (mean_distance_previous - mean_distance_current) + \
               (len(biggest_group_current) - len(biggest_group_previous)) + \
               reward_fish_eaten, self.result is not None
