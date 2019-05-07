"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    def __init__(self, nn_param_choices=None):
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    #create random set of nn_param_choices with different nb_neurons,nb_layers, activation, optimizer

    def create_set(self, network):
        self.network = network

    def train(self, dataset, path):
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset, path)

    def print_network(self):

        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
