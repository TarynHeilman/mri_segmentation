import unittest
import numpy as np
import pandas as pd
import os
import glob
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
        # make sure i and o directories have same files
        i_masks = [item.split('/')[-1] for item in glob.glob('{}/masks/*/i-contours/*.npy')]
        o_masks = [item.split('/')[-1] for item in glob.glob('{}/masks/*/o-contours/*.npy')]
        self.assertEqual(i_masks, o_masks)
        # make sure everything referenced on the class exists in files, visa versa
        mask_nums = [tup[1] for tup in pipe.mask_tuples]
        self.assertEqual(i_masks, mask_nums)

    def test_read_batch_arrays(self):
        pipe = p.Pipeline('/Users/taryn/projects/mri_segmentation/data')
        # very basic - check to see if X and Y have same shape
        sample_imgs = [pipe.format_impaths(pat, num) for pat, num in pipe.image_tuples[:10]]
        sample_masks = pipe.mask_tuples[:10]
        sample_i_masks = [pipe.format_maskpaths(orig, 'i', num) for orig, num in sample_masks]
        x, y1 = pipe.read_batch_arrays(sample_imgs, sample_i_masks)
        self.assertEqual(x.shape, y1.shape)
        # do this again for o contours
        sample_o_masks = [pipe.format_maskpaths(orig, 'o', num) for orig, num in sample_masks]
        x, y2 = pipe.read_batch_arrays(sample_imgs, sample_o_masks)
        self.assertEqual(x.shape, y2.shape)


if __name__ == '__main__':
    unittest.main()
