import os
import sys
import neat
#import visualize

import core
from multiprocessing import Lock
from threading import Thread
import gym
import gym_piranhas
import numpy as np
from time import time, sleep
import random
import pickle
# start the client-server communication
host = '127.0.0.1'
port = 13050

game_environments = []
client_creation_lock = Lock()

MEMORY_SIZE_PER_TURN = 1e2
game_states = {}
queued_game_states = []

GAME_STATE_MEMORY_FILEPATH = './game_state_memory.data'

def load_data(filepath, debug=True):
    if debug:
        print('Loading data from {}'.format(filepath))
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        if debug:
            print('Loaded data from {}'.format(filepath))
        return data

def store_data(data, filepath, debug=True):
    if debug:
        print('Storing data in {}'.format(filepath))
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)
        if debug:
            print('Stored data in {}'.format(filepath))

def get_inactive_environment():
    print('waiting for client_creation_lock')
    with client_creation_lock:

        print('got lock')
        for env in game_environments:
            if env.result is not None:
                # this game has ended
                print('reusing environment')
                return env

        env = gym.make('piranhas-v0')
        game_environments.append(env)

        print('created environment')
        return env

def evaluate_genome(genome, config):
    global game_states
    start_time = time()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    print('time taken: setup net', time() - start_time)

    start_time = time()
    env = get_inactive_environment()
    env.simulation_is_active = False
    observation = env.reset()
    reward = 0
    done = False
    userInfo = {}
    print('time taken: setup env', time() - start_time)


    genome.fitness = 0

    # now lets play a whole game
    while not done:
        start_time = time()
        action = net.activate(observation)
        print('time taken: forward pass', time() - start_time)
        # action has to be a number from 0 to 799 (10 * 10 * 8)
        action = np.array(action)
        action = np.argmax(action)
        # environment -> step
        previous_game_state = env.currentGameState
        start_time = time()
        observation, reward, done, userInfo = env.step(action)
        print('time taken: step', time() - start_time)
        print('->', reward, done, userInfo)

        queued_game_states.append((
            previous_game_state,
            action,
            env.currentGameState,
            reward,
            done
        ))

        # get reward
        genome.fitness += reward

    print('game over')

    number_of_turns = len(game_states)
    if number_of_turns > 0:
        print('training from memory')
        env.simulation_is_active = True
        additional_fitness = 0.

        number_of_trainings = 0
        for turn, game_states_for_turn in game_states.items():
            count = len(game_states_for_turn)
            number_of_trainings += count
            print('Turn {} - count {}:'.format(turn, count), end=' ')
            fitness_this_turn = 0.
            for i, (game_state, _, _, stored_reward, _) in enumerate(game_states_for_turn):
                if i % 10 == 0:
                    print('.', end='')
                    sys.stdout.flush()

                observation = env.reset(game_state)

                action = net.activate(observation)

                # action has to be a number from 0 to 799 (10 * 10 * 8)
                action = np.array(action)
                action = np.argmax(action)

                _, reward, done, _ = env.step(action)
                '''if reward > stored_reward:
                    game_states_for_turn[turn][i] = (
                        game_state,
                        action,
                        env.currentGameState,
                        reward,
                        done
                    )
                '''
                fitness_this_turn += reward
            fitness_this_turn /= float(count)
            print(' ->', fitness_this_turn)
            additional_fitness += fitness_this_turn

        genome.fitness += 9 * (additional_fitness / len(game_states))
        print('\ndone')

    return genome.fitness

def eval_genomes(genomes, config):
    global game_states, queued_game_states

    start_time = time()
    # new episode has begun
    # -> add all the experienced game_states to the training set
    for queued_game_state in queued_game_states:
        turn = queued_game_state[0].turn
        if turn not in game_states:
            game_states[turn] = []

        # TODO: filter for too many equal moves

        # don't let the memory take on arbitrary size
        count = len(game_states[turn])
        if count >= MEMORY_SIZE_PER_TURN:
            game_states[turn] = random.sample(game_states[turn], MEMORY_SIZE_PER_TURN - 1)

        # append the current game state
        game_states[turn].append(queued_game_state)

    queued_game_states = []
    print('time taken -> add queued game states:', time() - start_time)

    start_time = time()
    store_data(game_states, GAME_STATE_MEMORY_FILEPATH)
    print('time taken -> store queued game states:', time() - start_time)

    #evaluator = neat.threaded.ThreadedEvaluator(10, evaluate_genome)
    #evaluator.evaluate(genomes, config)

    for genome_id, genome in genomes:
        fitness = evaluate_genome(genome, config)

def run(config_file):
    global game_states, GAME_STATE_MEMORY_FILEPATH
    # Load configuration.
    '''
    start_time = time()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    print('time taken: load config', time() - start_time)
    # Create the population, which is the top-level object for a NEAT run.
    start_time = time()
    p = neat.Population(config)
    print('time taken: create population', time() - start_time)
    '''
    start_time = time()
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-18')
    print('time taken: restore checkpoint', time() - start_time)

    start_time = time()
    game_states = load_data(GAME_STATE_MEMORY_FILEPATH)
    print('time taken: restore replay memory', time() - start_time)


    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    print('run simulation')
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 3000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    fitness = evaluate_genome(genome, config)
    print('Fitness g_{id}: {fitness}'.format(
        id='winner',
        fitness=fitness
    ))

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-12')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config')
    run(config_path)
