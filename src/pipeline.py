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
        dct = parse_dicom_file(img_file)
        coords = parse_contour_file(contour_file)
        img = dct['pixel_data']

        mask = poly_to_mask(coords, *img.shape)

        # formulate file name, make sure directories exist, etc.
        source, typ, ext = contour_file.split('/')[-3:]
        # should use some kind of recursion here..
        sourcedir = subdir = '{}/{}'.format(self.maskdir, source)
        if not os.path.exists(sourcedir):
            os.mkdir(subdir)

        subdir = '{}/{}'.format(sourcedir, typ)
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # save as .npy for quick access
        fname = '{}/{}.npy'.format(subdir, ext.split('.')[0])
        np.save(fname, mask)

        # add target path to list
        if typ[0] == 'i':
            self.i_mask_paths.append(fname)
        elif typ[0] == 'o':
            self.o_mask_paths.append(fname)
        else:
            return('Unexpected file path')

    def create_masks(self):
        '''
        loops over all files to create boolean masks where both image and contour file exist

        :return: None
        '''
        # read in link.csv
        linked_files = pd.read_csv('{}/link.csv'.format(self.datadir))

        # instantiate lists of paths - store on class for help at runtime
        # keep lists separate for i and o images for ease of analysis
        self.i_image_paths, self.o_image_paths, self.i_mask_paths, self.o_mask_paths = [], [], [], []

        # loop over subdirectories
        for img, orig in linked_files.itertuples(index=False):
            # get list of all files in these directories
            img_files = glob.glob(self.datadir+'/dicoms/'+img+'/*.dcm')

            # get number to match with contour files
            img_nums = [int(file.split('/')[-1][:-4]) for file in img_files]

            for num, img_file in zip(img_nums, img_files):
                # format (possible) corresponding contour file
                i_cfile, o_cfile = ('{0}/contourfiles/{1}/{2}-contours/IM-0001-{3}-{2}contour-manual.txt'.format(
                    self.datadir,
                    orig,
                    ctype,
                    str(num).zfill(4)) for ctype in ('i', 'o'))

                if os.path.isfile(i_cfile):
                    self.process_one_mask(img_file, i_cfile)
                    # add image paths to list
                    self.i_image_paths.append(img_file)

                if os.path.isfile(o_cfile):
                    self.process_one_mask(img_file, o_cfile)
                    # add image paths to list
                    self.o_image_paths.append(img_file)

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
