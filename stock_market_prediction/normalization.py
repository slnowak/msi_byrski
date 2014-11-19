import numpy
from pybrain.datasets.supervised import SupervisedDataSet

__author__ = 'novy'


class Normalizer(object):
    def normalize(self, data):
        return self._subtract_mean_and_divide_by_range(data)

    def _subtract_mean_and_divide_by_range(self, array):
        return (array - array.mean(axis=0)) / (
            array.max(axis=0) - array.min(axis=0))


class DataSetCreator(object):

    def create_datasets(self, data, normalizer):
        normalized_data = normalizer.normalize(data)
        training_data, test_data = normalized_data[:-len(normalized_data)/4], normalized_data[-len(normalized_data)/4:]
        return self._filled_dataset(training_data), self._filled_dataset(test_data)

    def _filled_dataset(self, normalized_data):
        data_set = SupervisedDataSet(normalized_data.shape[1] - 1, 1)
        for normalized_row in normalized_data:
            data_set.addSample(normalized_row[:-1].tolist(), normalized_row[-1])

        return data_set


def create_normalized_datasets(data):
    return DataSetCreator().create_datasets(data, Normalizer())
