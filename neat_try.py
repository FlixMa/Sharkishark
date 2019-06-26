import os
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

MEMORY_SIZE_PER_TURN = 1e3
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

    number_of_trainings = len(game_states)
    if number_of_trainings > 0:
        print('training from memory')
        env.simulation_is_active = True
        additional_fitness = 0.

        for turn, game_states_for_turn in game_states.items():
            print('Turn {}:'.format(turn), end=' ')
            for i, (game_state, _, _, stored_reward, _) in enumerate(game_states_for_turn):
                if i % 10 == 0:
                    print('.', end='' if i % 80 != 0 else '\n')

                observation = env.reset(game_state)

                action = net.activate(observation)

                # action has to be a number from 0 to 799 (10 * 10 * 8)
                action = np.array(action)
                action = np.argmax(action)

                _, reward, done, _ = env.step(action)
                if reward > stored_reward:
                    game_states_for_turn[turn][i] = (
                        game_state,
                        action,
                        env.currentGameState,
                        reward,
                        done
                    )
                additional_fitness += reward

        genome.fitness += 9 * (additional_fitness / number_of_trainings)
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

    threads = []
    for genome_id, genome in genomes:

        def job():
            print('Thread for genome {id} started.'.format(
                id=genome_id
            ))
            fitness = evaluate_genome(genome, config)

            print('Fitness of genome {id}: {fitness}'.format(
                id=genome_id,
                fitness=fitness
            ))
        job()

    '''
        thread = Thread(target=job)
        threads.append(thread)
        thread.name = 'Fred'
        thread.start()
        sleep(5)

    # wait for all genomes to be tested
    time_started_waiting = time()
    for thread in threads:
        time_remaining = 11.0 - (time() - time_started_waiting)
        if time_remaining > 0:
            # wait for the thread to be done
            # after 11 seconds just dont care about it anymore
            # and just kill it next time we need an environment
            thread.join(time_remaining)
    '''

def run(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    print('config loaded')
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    print('population created')

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-12')
    print('checkpoint restored')

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
