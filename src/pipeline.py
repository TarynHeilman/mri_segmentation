import numpy as np
import pandas as pd
import glob
import pydicom
from pydicom.errors import InvalidDicomError
import os
from src.parsing import *

class Pipeline(object):
    def __init__(self, datadir, compute_masks = False):
        '''
        add some notes in init if needed
        '''
        self.datadir = datadir
        self.maskdir = '{}/masks'.format(self.datadir)
        if os.listdir(self.maskdir) == [] or compute_masks:
            self.create_masks()

    def process_one_mask(self, img_file, contour_file):
        '''
        creates one boolean mask

        :param img_file: str, path to image file
        :param contour_file: str, path to contour file
        :return: None or np array
        '''
        dct = parse_dicom_file(img_file)
        coords = parse_contour_file(contour_file)
        img = dct['pixel_data']

        mask = poly_to_mask(coords, *img.shape)

        # save image as binary for quick access
        fname = self.maskdir+contour_file.split('/')[-1]
        mask.tofile(fname)

    def create_masks(self):
        '''
        loops over all files to create boolean masks where both image and contour file exist

        :return: None or np array
        '''
        # loop over subdirectories
        for img, orig in linked_files.itertuples(index=False):
            img_files = glob.glob(self.datadir+'/dicoms/'+img+'/*.dcm')

            # get number to match with contour files
            img_nums = [int(file.split('/')[-1][:-4]) for file in img_files]

            for num, img_file in zip(img_nums, img_files):
                # format (possible) corresponding contour file
                cfile = '{}/contourfiles/{}/i-contours/IM-0001-{}-icontour-manual.txt'.format(
                    self.datadir,
                    orig,
                    str(num).zfill(4))

                if os.path.isfile(cfile):
                    self.process_one(img_file, cfile)


def parse():
    '''
    argparser for command line arguments

    :return: tuple of argparse.Namespace
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--path-to-data',
      help = 'Where the data is stored',
      default = '/Users/taryn/projects/mri_segmentation/data')

    return parser.parse_known_args()


if __name__ == "__main__":
    args = parse()
    datadir = args[0].path_to_data

    pipeline = Pipeline(datadir)
