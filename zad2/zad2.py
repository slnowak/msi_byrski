from math import sin, pi
from numpy.core.numeric import arange
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

__author__ = 'novy'


def create_dataset():
    dataset = SupervisedDataSet(1, 1)

    for x in arange(0, 4*pi, pi/30):
        dataset.addSample(x, sin(x))

    return dataset


def train(net, data_set):
    trainer = BackpropTrainer(net, data_set, learningrate=0.04, momentum=0.5, verbose=True)
    trainer.trainUntilConvergence(dataset=data_set, maxEpochs=5000)


dataset = create_dataset()
net = buildNetwork(1, 20, 1)
train(net, dataset)

print 'sin 0 = ' + str(net.activate([0])[0])
print 'sin PI/2 = ' + str(net.activate([pi/2])[0])
print 'sin PI/6 = ' + str(net.activate([pi/6])[0])

