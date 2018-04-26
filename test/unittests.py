import unittest
import numpy as np
import pandas as pd
from src.parsing import *
from src import pipeline as p

class TestPipeline(unittest.TestCase):

    def test_poly_to_mask(self):
        square_coords = [(1, 1), (1, 3), (3, 3), (3, 1)]
        mask = poly_to_mask(square_coords, 5, 5)
        test_result = np.array([[False]*5,
                                [False]*5,
                                [False, False, True, False, False],
                                [False]*5,
                                [False]*5])
        self.assertTrue((mask == test_result).all())


if __name__ == '__main__':
    unittest.main()
