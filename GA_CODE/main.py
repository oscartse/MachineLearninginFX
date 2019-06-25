"""Entry point to evolving the neural network. Start here."""
import logging
import sys
from optimizer import Optimizer
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='EURUSDlog.txt'
)

def train_networks(networks, dataset, path):

    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, path)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset, path):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    #getdata = train.getdata(path)

    # Evolve the generation.
    for i in range(generations):
        logging.info("**Doing generation %d of %d**" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset, path)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('='*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""

    generations = 5  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = 'EURUSD60'
    path = '/Users/oscartse/Desktop/FXFYP-master/GA_CODE/POCtestdata/EURUSD60.csv'
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4, 5],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
        'dropout_rate': [0.8, 0.5, 0.3, 0.1],
    }
    logging.info("**Evolving %d generations with population %d**" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset, path)

if __name__ == '__main__':
    main()
