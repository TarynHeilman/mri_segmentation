import unittest
import numpy as np
import pandas as pd
import os
import glob
from src.parsing import *
from src import pipeline as p
from src import analysis as a

class TestPipeline(unittest.TestCase):

    def test_outer_exclusionary(self):
        img = np.arange(1, 37).reshape(6, 6)
        i_mask = np.array([[False]*6,
                           [False]*6,
                           [False, False, True, True, False, False],
                           [False, False, True, True, False, False],
                           [False]*6,
                           [False]*6])
        o_mask = np.array([[False]*6,
                           [False, True, True, True, True, False],
                           [False, True, True, True, True, False],
                           [False, True, True, True, True, False],
                           [False, True, True, True, True, False],
                           [False]*6])
        result = a.avg_pixel_value_outer(img, i_mask, o_mask)
        self.assertEqual(18.5, result)



if __name__ == '__main__':
    unittest.main()
