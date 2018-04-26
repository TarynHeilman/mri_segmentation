import unittest
import numpy as np
import pandas as pd
import os
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

    def test_create_masks(self):
        pipe = p.Pipeline('/Users/taryn/projects/mri_segmentation/data')
        # very basic - check to make sure data and targets are same length!
        self.assertEqual(len(pipe.target_paths), len(pipe.image_paths))

    def test_read_batch_arrays(self):
        pipe = p.Pipeline('/Users/taryn/projects/mri_segmentation/data')
        # very basic - check to see if X and Y have same shape
        sample_imgs, sample_targets = pipe.image_paths[:10], pipe.target_paths[:10]
        x, y = pipe.read_batch_arrays(sample_imgs, sample_targets)
        self.assertEqual(x.shape, y.shape)        

if __name__ == '__main__':
    unittest.main()
