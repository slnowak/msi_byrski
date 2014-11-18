from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

__author__ = 'novy'


class TrainingExecutor(object):
    def __init__(self, training_dataset, test_dataset, epochs, learning_rate, momentum):
        super(TrainingExecutor, self).__init__()
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.hidden_layers = 20

    def get_trained_network(self):
        network = buildNetwork(self.training_dataset.indim,  self.hidden_layers,
                               self.training_dataset.outdim, recurrent=True)

        trainer = BackpropTrainer(network, learningrate=self.learning_rate, momentum=self.momentum)

        self.train_network(trainer)
        return network

    def train_network(self, trainer):

        for _ in xrange(0, self.epochs):
            trainer.trainOnDataset(self.training_dataset, 1)
            trainer.testOnData(self.test_dataset)
