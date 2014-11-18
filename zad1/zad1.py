from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer


def create_dataset():
    data = SupervisedDataSet(2, 1)

    data.addSample([1, 1], [0])
    data.addSample([1, 0], [1])
    data.addSample([0, 1], [1])
    data.addSample([0, 0], [0])

    return data


def train(net, data_set):
    trainer = BackpropTrainer(net, data_set, learningrate=0.01, momentum=0.99, verbose=False)
    for epoch in xrange(0, 1000):
        trainer.train()


data_set = create_dataset()
net = buildNetwork(2, 3, 1)
train(net, data_set)

print 'xor 0 1 ' + str(net.activate([0, 1])[0])
print 'xor 1 0 ' + str(net.activate([1, 0])[0])
print 'xor 0 0 ' + str(net.activate([0, 0])[0])
print 'xor 1 1 ' + str(net.activate([1, 1])[0])
print '----------------------------------------'
print 'xor 0.001 0.1 ' + str(net.activate([0.001, 0.1])[0])
print 'xor 0.1 0.9 ' + str(net.activate([0.1, 0.9])[0])