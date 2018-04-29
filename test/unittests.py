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
        # make sure they reference the same numbers - messy
        i_img_nums = [int(fil.split('/')[-1][:-4]) for fil in pipe.i_image_paths]
        i_mask_nums = [int(fil.split('/')[-1].split('-')[2]) for fil in pipe.i_mask_paths]
        self.assertEqual(set(i_img_nums), set(i_mask_nums))
        o_img_nums = [int(fil.split('/')[-1][:-4]) for fil in pipe.o_image_paths]
        o_mask_nums = [int(fil.split('/')[-1].split('-')[2]) for fil in pipe.o_mask_paths]
        self.assertEqual(set(o_img_nums), set(o_mask_nums))

    def test_read_batch_arrays(self):
        pipe = p.Pipeline('/Users/taryn/projects/mri_segmentation/data')
        # very basic - check to see if X and Y have same shape
        i_sample_imgs, i_sample_targets = pipe.i_image_paths[:10], pipe.i_mask_paths[:10]
        x1, y1 = pipe.read_batch_arrays(i_sample_imgs, i_sample_targets)
        self.assertEqual(x1.shape, y1.shape)
        # do this again for o contours
        o_sample_imgs, o_sample_targets = pipe.o_image_paths[:10], pipe.o_mask_paths[:10]
        x2, y2 = pipe.read_batch_arrays(o_sample_imgs, o_sample_targets)
        self.assertEqual(x2.shape, y2.shape)

if __name__ == '__main__':
    unittest.main()
