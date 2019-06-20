#!/usr/bin/env python
# coding: utf-8

import keras
import gym
import gym_piranhas
import numpy as np

env = gym.make('piranhas-v0')
np.random.seed(12342)
env.seed(123042)
nb_actions = env.action_space.shape[0]

from keras.models import Sequential, Model
from keras.layers import Convolution2D, Convolution3D, Input, Concatenate, Flatten, Dense, Activation, Permute
from keras.optimizers import Adam
import keras.backend as K

INPUT_SHAPE = 10
DIMENSIONS  = 33
# WINDOW_LENGTH = 4  # The number of past states we consider
input_shape = (DIMENSIONS, INPUT_SHAPE, INPUT_SHAPE)  # (WINDOW_LENGTH, INPUT_SHAPE, INPUT_SHAPE, DIMENSIONS)

actor = Sequential()
if K.image_data_format() == 'channels_first':
    # (batch, width, height, channels)
    actor.add(Permute((3, 1, 2), input_shape=input_shape))
elif K.image_data_format() == 'channels_last':
    # (batch, channels, width, height)
    actor.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering')

# The actual network structure
# results in a (6 x 6 x 32) output volume
actor.add(Convolution2D(64, (1, 1), activation='relu', data_format='channels_first'))
# results in a (4 x 4 x 64) output volume
actor.add(Convolution2D(128, (3, 3), activation='relu', data_format='channels_first'))
# results in a (2 x 2 x 64) output volume
actor.add(Convolution2D(128, (1, 1), activation='relu', data_format='channels_first'))
# flattens the result (vector of size 256)
actor.add(Flatten())
actor.add(Dense(1024, activation='relu'))
# add fully-connected layer
actor.add(Dense(512, activation='relu'))
actor.add(Dense(256, activation='relu'))
# map to output: coordinates and move
actor.add(Dense(nb_actions, activation='linear'))
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=input_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.random import OrnsteinUhlenbeckProcess


class PiranhasProcessor(Processor):

    def process_state_batch(self, batch):
        """
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        """
        return np.squeeze(batch, axis=1)

    def process_reward(self, reward):
        return np.clip(reward, -10., 10.)

# copied from https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
# Add memory


# https://stackoverflow.com/questions/47140642/what-does-the-episodeparametermemory-of-keras-rl-do
# for explanation of parameters
memory = SequentialMemory(limit=1000000, window_length=1)  # TODO WINDOW_LENGTH

# preprocessing is done in onGameStateUpdate in the environment
# this processor removes the unnecessary dimension created by keras-rl
processor = PiranhasProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  processor=processor, random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

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

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import os

mode = 'train'
if mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(env.name)
    checkpoint_weights_filename = 'dqn_' + env.name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env.name)
    # Add a checkpoint every 250000 iterations
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=25000)]
    # log to file every 100 iterations
    callbacks += [FileLogger(log_filename, interval=100)]
    if os.path.exists(weights_filename):
        agent.load_weights(weights_filename)
    agent.fit(env, callbacks=callbacks, nb_steps=1000000000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    agent.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    agent.test(env, nb_episodes=10, visualize=False)
elif mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(env.name)
    agent.load_weights(weights_filename)
    agent.test(env, nb_episodes=10, visualize=True)
