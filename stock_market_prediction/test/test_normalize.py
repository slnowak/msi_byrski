from unittest import TestCase
import numpy
from numpy.testing.utils import assert_allclose
from stock_market_prediction.normalization import Normalizer

__author__ = 'novy'


class TestNormalize(TestCase):
    def test_normalize(self):

        normalizator = Normalizer()
        mean = lambda l: sum(l) / float(len(l))

        array = numpy.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 2, 1]
            ]
        )

        expected_result = numpy.array(
            [
                [(1 - mean((1, 4, 7))) / 6, (2 - mean((2, 5, 2))) / 3, 3],
                [(4 - mean((1, 4, 7))) / 6, (5 - mean((2, 5, 2))) / 3, 6],
                [(7 - mean((1, 4, 7))) / 6, (2 - mean((2, 5, 2))) / 3, 1]
            ]
        )

        assert_allclose(normalizator.normalize(array), expected_result)