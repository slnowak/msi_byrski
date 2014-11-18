import numpy

__author__ = 'novy'


class DataReader(object):
    def __init__(self):
        super(DataReader, self).__init__()

    def read_data(self, filepath):

        with open(filepath, 'r') as input_file:
            result = [
                [float(x) for x in line.split()] for line in input_file.readlines()
            ]

        return numpy.array(result)