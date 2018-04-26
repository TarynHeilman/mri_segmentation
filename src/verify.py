import numpy as np
import matplotlib.pyplot as plt
import argparse

from src.parsing import *

def visualize(img_file, contour_file, plotname = None):
    '''
    visualize output of parsing functions

    :param img_file: str, path to image file
    :param contour_file: str, path to contour file
    :param plotname: optional, str, path to save generated plot
    :return: None
    '''
    dct = parse_dicom_file(img_file)
    coords = parse_contour_file(contour_file)
    img = dct['pixel_data']

    mask = poly_to_mask(coords, *img.shape)

    fig, axarr = plt.subplots(nrows = 1, ncols = 2)

    axarr[0].imshow(img)
    axarr[0].set_title('Image')

    axarr[1].imshow(mask)
    axarr[1].set_title('Mask')

    if plotname:
        plt.savefig(plotname)
    else:
        plt.show()



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

    parser.add_argument(
      '--plotname',
      help = 'Where to save the plot, if applicable',
      default = None)

    return parser.parse_known_args()


if __name__ == '__main__':
    args = parse()

    datadir = args[0].path_to_data
    plotname = args[0].plotname

    visualize(datadir+'dicoms/SCD0000101/148.dcm', datadir+'contourfiles/SC-HF-I-1/i-contours/IM-0001-0148-icontour-manual.txt', plotname = plotname)
