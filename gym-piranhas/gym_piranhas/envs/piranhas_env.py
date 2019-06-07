import gym
from gym import error, spaces, utils
from gym.utils import seeding


class PiranhasEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.action_space = spaces.Dict({
			"fish": spaces.Tuple((spaces.Discrete(10), spaces.Discrete(10))),
			"direction": spaces.Discrete(8),
		})

		self.observation_space = spaces.Dict({
			"board": spaces.Box(
				low=0, high=3, shape=(10, 10), dtype=int
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
			"board": [[0,1,1,1,1,1,1,1,1,0],
						[2,0,0,0,0,0,0,0,0,2],
						[2,0,0,0,0,0,0,0,0,2],
						[2,0,0,3,0,0,0,0,0,2],
						[2,0,0,0,0,0,0,0,0,2],
						[2,0,0,0,0,0,0,0,0,2],
						[2,0,0,0,0,0,0,0,0,2],
						[2,0,0,0,0,0,0,3,0,2],
						[2,0,0,0,0,0,0,0,0,2],
					    [0,1,1,1,1,1,1,1,1,0]],
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

		self.name = 'piranhas'

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

	# def apply_action(self, action):


	def step(self, action):
		"""
		Run one timestep of the environment's dynamics. When end of
	    episode is reached, you are responsible for calling `reset()`
	    to reset this environment's state.
	    Accepts an action and returns a tuple (observation, reward, done, info).
	    Args:
	 	    action (object): an action provided by the agent
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



		assert self.action_space.contains(action)
		self.observation = apply_action(action)

		reward = 0
		if self.observation["opponent"]["nr_fish"] > \
				self.observation["me"]["nr_fish"]:
			reward = -1
		elif self.observation["opponent"]["nr_fish"] < \
				self.observation["me"]["nr_fish"]:
			reward = 1

		return self.observation, reward, False


	def reset(self):
		"""
		Resets the state of the environment and returns an initial observation.
		Returns:
			observation (object): the initial observation.
		"""
		pass

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
