import numpy as np
import pandas as pd
import argparse
import glob
import pydicom
from pydicom.errors import InvalidDicomError
import os
from src.parsing import *
from sklearn.model_selection import KFold

class Pipeline(object):
    def __init__(self, datadir, verbose = False, batch_size = 8):
        '''
        add some notes in init if needed
        '''
        self.datadir = datadir
        self.maskdir = '{}/masks'.format(self.datadir)
        self.verbose = verbose
        self.batch_size = batch_size

        self.read_links()

        if self.verbose:
            print('computing target masks...')
        self.create_masks()

    def read_links(self):
        '''
        helper function, called in init, stores links (dict) on the class
        '''
        # read in link.csv
        linked_files = pd.read_csv('{}/link.csv'.format(self.datadir))

        self.links = {a : b for a, b in linked_files.itertuples(index = False)}

    def format_impaths(self, source, num):
        '''
        helper function to format image filepaths from the given source and number

        :param source: str
        :param num: int

        return: impath (str)
        '''
        return '{}/dicoms/{}/{}.dcm'.format(self.datadir, source, num)

    def format_maskpaths(self, orig, typ, num):
        '''
        helper function to format mask filepaths from the given origination and number.
        masks are named with the same convention as images
        makes directories if required

        :param orig: str
        :param typ: str (i or o)
        :param num: int

        return: maskpath (str)
        '''
        subdir = '{}/{}/{}-contours'.format(self.maskdir, orig, typ)
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        return '{}/{}.npy'.format(subdir, num)

    def format_contourpaths(self, orig, typ, num):
        '''
        helper function to format contour filepaths from the given origination and number

        :param orig: str
        :param typ: str (i or o)
        :param num: int

        return: maskpath (str)
        '''
        return '{0}/contourfiles/{1}/{2}-contours/IM-0001-{3}-{2}contour-manual.txt'.format(
                self.datadir,
                orig,
                typ,
                str(num).zfill(4))

    def generate_image_tuples(self):
        '''
        helper function, loops over all image directories and returns a list of \
        tuples referencing unique identifiers

        :return: img_tups (list)
        '''
        img_tups = []
        # just need patient images in this case
        for img in self.links:
            # get list of all files in these directories
            img_files = glob.glob(self.datadir+'/dicoms/'+img+'/*.dcm')

            # get number to match with contour files
            img_nums = [int(file.split('/')[-1][:-4]) for file in img_files]

            img_tups.append((img, img_nums))

        return img_tups

    def process_one_mask(self, img_file, contour_file):
        '''
        creates one boolean mask

        :param img_num: str, path to image file
        :param contour_file: str, path to contour file
        :return: mask (np array)
        '''
        dct = parse_dicom_file(img_file)
        coords = parse_contour_file(contour_file)
        img = dct['pixel_data']

        mask = poly_to_mask(coords, *img.shape)

        return mask

    def create_masks(self):
        '''
        loops over all files to create boolean masks where both image and contour file exist

        :return: None
        '''
        # store tuples with origination/source and image # on class for easier access
        self.image_tuples, self.mask_tuples = [], []

        img_tups = self.generate_image_tuples()

        for pat, num in img_tups:
            # get matching origination
            orig = self.links[pat]

            # format (possible) corresponding contour files
            cfiles = [self.format_contourpaths(orig, typ, num) for typ in ('i', 'o')]

            # only process when both inner and outer exist
            if os.path.isfile(cfiles[0]) and os.path.isfile(cfiles[1]):

                for cfil, typ in zip(cfiles, ('i', 'o')):
                    mask = self.process_one_mask(img_file, cfil)
                    fname = self.format_maskpaths(orig, typ, num)
                    np.save(fname, imask)

                # add identifiers to list
                self.image_tuples.append((pat, num))
                self.mask_tuples.append((orig, num))

    def read_batch_arrays(self, X_files, Y_files):
        '''
        helper function that reads in lists of files

        :param X_files: list or array like, paths to image files
        :param Y_files: list or array like, paths to mask files
        :return: X (np array), Y (np array)
        '''
        X, Y = [], []
        for xfile, yfile in zip(X_files, Y_files):
            x_data = parse_dicom_file(xfile)['pixel_data']
            X.append(x_data)
            Y.append(np.load(yfile))

        return np.array(X), np.array(Y)


    def batch_train(self):
        '''
        loops over all files to create boolean masks where both image and contour file exist

        :return: None
        '''
        # get random indices to match file paths
        dim = len(self.image_paths)
        kf = KFold(n_splits = dim//self.batch_size + 1, shuffle = True)

        for x_files, y_files in kf.split():
            X, Y = self.read_batch_arrays(x_files, y_files)

            # at this point, the data are going into some model that I am not privy to
            # some_model_training(X, Y)


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
    print(pipeline.target_paths)
