import numpy
from stock_market_prediction.data_reader import DataReader
from stock_market_prediction.normalization import create_normalized_datasets
from stock_market_prediction.train import TrainingExecutor
import matplotlib.pyplot as plt

__author__ = 'novy'


def generate_predictions_series(network, dataset):
    return [network.activate(row)[0] for row in dataset['input']]


def train_network(training_dataset, test_dataset, epochs=10, learning_rate=0.01, momentum=0.98):
    executor = TrainingExecutor(training_dataset, test_dataset, epochs, learning_rate, momentum)
    iterations = 30

    predictions = []
    for _ in xrange(iterations):
        network = executor.get_trained_network()
        predictions.append(
            generate_predictions_series(network, test_dataset)
        )

    predictions = numpy.array(predictions)
    mean = predictions.mean(axis=0).tolist()
    minimum = predictions.min(axis=0).tolist()
    maximum = predictions.max(axis=0).tolist()
    real = reduce(lambda x, y: x+y, test_dataset['target'].tolist())

    x = [i for i in range(1, len(real) + 1)]
    plt.title(("Liczba epok: %d, szybkosc uczenia: %f, momentum: %f" % (epochs, learning_rate, momentum)))
    plt.plot(x, real, label="Wartosci rzeczywiste", color='red')
    plt.plot(x, mean, label="Wartosci srednie", color='green')
    plt.plot(x, minimum, label="Wartosci min", color='grey')
    plt.plot(x, maximum, label="Wartosci max", color='grey')
    plt.legend()
    plt.show()


def create_output_string(data):
    return '\n'.join([' '.join([str(number) for number in row]) for row in data])


def write_to_file(filepath, data):
    with open(filepath, 'wb') as target:
        target.write(
            create_output_string(data)
        )

if __name__ == '__main__':
    raw_data = DataReader().read_data('input')
    training_dataset, test_dataset = create_normalized_datasets(raw_data)
    train_network(training_dataset, test_dataset, epochs=10, learning_rate=0.01, momentum=0.01)

