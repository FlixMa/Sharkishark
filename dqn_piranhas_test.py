#!/usr/local/bin/python3
# coding: utf-8

import keras
import gym
import gym_piranhas
import numpy as np

env = gym.make('piranhas-v0')
np.random.seed(12342)
env.seed(123042)
nb_actions = env.action_space.n

from keras.models import Sequential
from keras.layers import Convolution2D, Convolution3D, Flatten, Dense, Activation, Permute
from keras.optimizers import Adam
import keras.backend as K

model = Sequential()
model.add(Dense(400, input_shape=env.observation_space.shape, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
# map to output: coordinates and move
model.add(Dense(nb_actions, activation='linear'))
model.build()
print(model.summary())


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from core.state import GameState
from core.util import FieldState, PlayerColor


class PiranhasProcessor(Processor):

    def process_state_batch(self, batch):
        """
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        """
        return np.squeeze(batch, axis=1)

    def process_reward(self, reward):
        with open('./rewards.txt', 'a') as file:
            file.write('%.3f\n' % float(reward))
        return np.clip(reward, -100., 100.)

# copied from https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
# Add memory


# https://stackoverflow.com/questions/47140642/what-does-the-episodeparametermemory-of-keras-rl-do
# for explanation of parameters
memory = SequentialMemory(limit=10000, window_length=1)  # TODO WINDOW_LENGTH

# preprocessing is done in onGameStateUpdate in the environment
# this processor removes the unnecessary dimension created by keras-rl
processor = PiranhasProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = EpsGreedyQPolicy(0.9)

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    processor=processor,
    gamma=.9, # discounting factor
    delta_clip=1.0,
    batch_size=32,
    nb_steps_warmup=512
    #enable_dueling_network=True,
    #dueling_type='avg'
)
dqn.compile(Adam(lr=0.025), metrics=['mae'])

# start server before this

import core
import subprocess

# start the client-server communication
host = '127.0.0.1'
port = 13050

game_client = None


def makeNewGame():
    global game_client, host, port

    if game_client is not None:
        game_client.stop()

    game_client = core.communication.GameClient(host, port, env)

    game_client.start()  # connect to the server
    game_client.join()  # join a game

    # start the opponent

    # Use simple client
    subprocess.Popen("java -jar ../simpleclient-piranhas-19.2.1.jar --port {}".format(port),
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL,
                     shell=True)

    """
    # Use own client
    game_client2 = core.communication.GameClient(host, port, env)

    game_client2.start()  # connect to the server
    game_client2.join()  # join a game
    """


env.set_reset_callback(makeNewGame)

import os
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

mode = 'train'
if mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(env.name)
    checkpoint_weights_filename = 'dqn_' + env.name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env.name)
    #if os.path.exists(weights_filename):
    #    dqn.load_weights(weights_filename)
    # Add a checkpoint every 250000 iterations
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    # log to file every 100 iterations
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=17500000, log_interval=5000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(env.name)
    #if args.weights:
    #    weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
