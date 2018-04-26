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

        if self.verbose:
            print('computing target masks...')
        self.create_masks()

    def process_one_mask(self, img_file, contour_file):
        '''
        creates one boolean mask

        :param img_file: str, path to image file
        :param contour_file: str, path to contour file
        :return: None or np array
        '''
        # add image and contour paths to list
        self.image_paths.append(img_file)
        self.contour_paths.append(contour_file)

        dct = parse_dicom_file(img_file)
        coords = parse_contour_file(contour_file)
        img = dct['pixel_data']

        mask = poly_to_mask(coords, *img.shape)

        # formulate file name, make sure directories exist, etc.
        folder, typ, ext = contour_file.split('/')[-3:]
        subdir = '{}/{}'.format(self.maskdir, folder)
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # save as .npy for quick access
        fname = '{}/{}.npy'.format(subdir, ext.split('.')[0])
        np.save(fname, mask)

        # add target path to list
        self.target_paths.append(fname)

    def create_masks(self):
        '''
        loops over all files to create boolean masks where both image and contour file exist

        :return: None
        '''
        # read in link.csv
        linked_files = pd.read_csv('{}/link.csv'.format(self.datadir))

        # instantiate lists of paths - store on class for help at runtime
        self.image_paths, self.contour_paths, self.target_paths = [], [], []

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
                    self.process_one_mask(img_file, cfile)

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
