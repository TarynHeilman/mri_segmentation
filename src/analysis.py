import numpy as np
import argparse
from src.parsing import *
from src.pipeline import Pipeline, parse
import matplotlib.pyplot as plt
import scipy.stats as scs

def preprocess():
    '''
    creates masks with pipeline, reads in the matching images, and returns image arrays for analysis
    :return: images, i_masks, o_masks (np arrays), p (Pipeline object)
    '''
    args = parse()

    p = Pipeline(args[0].path_to_data)

    # get file-lists
    image_paths = [p.format_impaths(pat, num) for pat, num in p.image_tuples]
    i_mask_paths = [p.format_maskpaths(orig, 'i', num) for orig, num in p.mask_tuples]
    o_mask_paths = [p.format_maskpaths(orig, 'o', num) for orig, num in p.mask_tuples]

    images, i_masks = p.read_batch_arrays(image_paths, i_mask_paths)
    images, o_masks = p.read_batch_arrays(image_paths, o_mask_paths)

    return images, i_masks, o_masks, p

def avg_pixel_value_inner(img, mask):
    '''
    returns the average pixel value interior to the given mask

    :param img: np array
    :param mask: np array of booleans
    :return: inner_avg (float)
    '''
    return img[mask].mean()


def avg_pixel_value_outer(img, i_mask, o_mask):
    '''
    returns the average pixel value between the given masks

    :param img: np array
    :param i_mask: np array of booleans
    :param o_mask: np array of booleans
    :return: outer_avg (float)
    '''
    return img[o_mask * ~i_mask].mean()


def compute_avg_pixel_values(images, i_masks, o_masks):
    '''
    computes average inner and outer pixel values for each image and returns in two lists

    :param images: 3D np array (array of images)
    :param i_masks: 3D np array of booleans
    :param o_masks: 3D np array of booleans
    :return: inner_avg, outer_avg (list of floats)
    '''
    inner_avg, outer_avg = [], []
    for image, i, o in zip(images, i_masks, o_masks):
        inner_avg.append(avg_pixel_value_inner(image, i))
        outer_avg.append(avg_pixel_value_outer(image, i, o))

    return inner_avg, outer_avg


def plot_hists(inner_avg, outer_avg, plotname = None):
    '''
    plots histograms with inner and outer average pixel values

    :param inner_avg: (list of floats)
    :param outer_avg (list of floats)
    :return: None
    '''
    plt.clf()
    plt.hist(inner_avg, alpha = 0.5, normed = True, label = 'Inner Contour Averages')
    plt.hist(outer_avg, alpha = 0.5, color = 'red', normed = True, label = 'Outer Contour Averages')
    plt.legend()
    plt.title('Average pixel values inside contours')

    if plotname:
        plt.savefig(plotname)
    else:
        plt.show()


def threshold_mask(img, thresh, o_contour):
    '''
    computes a mask based on simple thresholds

    :param img: 2D np array
    :param thresh: float
    :return: pred_mask (2D np array of booleans)
    '''
    return (img >= thresh)*o_contour


def compute_2D_metrics(true_mask, pred_mask, outer_mask):
    '''
    plots histograms with inner and outer average pixel values

    :param true_mask: 2D np array
    :param outer_avg: 2D np array
    :return: (pixel-wise) accuracy, recall, precision, false positive rate
    '''
    TP = (true_mask*pred_mask*outer_mask).sum()
    FP = (~true_mask*pred_mask*outer_mask).sum()
    FN = (true_mask*~pred_mask*outer_mask).sum()
    TN = (~true_mask*~pred_mask*outer_mask).sum()

    acc = (TP + TN)/(outer_mask.sum())
    rec = TP / (TP + FN)
    prec = TP / (TP + FP)
    fpr = FP / (FP + TN)

    return acc, rec, prec, fpr


def plot_thresh_metrics(thresh_list, A, R, P):
    '''
    plots accuracy, precision, and recall for each threshold

    :param thresh_list: list of thresholds
    :param A: list of accuracies
    :param R: list of recalls
    :param P: list of precisions
    :return: None
    '''
    plt.clf()
    plt.plot(thresh_list, A, label = 'accuracy')
    plt.plot(thresh_list, R, label = 'recall')
    plt.plot(thresh_list, P, label = 'precision')
    plt.xlabel('Pixel Value Threshold')
    plt.title('Pixel Wise Classification Metrics')
    plt.legend()
    plt.savefig('images/threshold_metrics.png')


def plot_roc(thresh_list, R, FPR):
    '''
    plots roc curve
    :param R: list of recalls
    :param A: list of false positive rates
    :return: None
    '''
    plt.clf()
    plt.plot(FPR, R)
    z = np.linspace(0,1)
    plt.plot(z, z, ls='dotted')    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pixel-wise Receiver Operating Characteristic Curve')
    plt.savefig('images/roc.png')


def threshold_stats(images, i_masks, o_masks, thresh_list = np.arange(135, 345, 10)):
    # lots of for loops - gross - definitely can broadcast
    A, R, P, FPR = [], [], [], []
    for thresh in thresh_list:
        a, r, p, fpr = [], [], [], []
        for img, i, o in zip(images, i_masks, o_masks):
            pred_mask = threshold_mask(img, thresh, o)
            acc, rec, prec, fp = compute_2D_metrics(i, pred_mask, o)
            a.append(acc)
            r.append(rec)
            p.append(prec)
            fpr.append(fp)
        A.append(np.nanmean(a))
        R.append(np.nanmean(r))
        P.append(np.nanmean(p))
        FPR.append(np.nanmean(fpr))

    plot_thresh_metrics(thresh_list, A, R, P)
    plot_roc(thresh_list, R, FPR)


if __name__ == '__main__':
    images, i_masks, o_masks, p = preprocess()
    # inner_avg, outer_avg = compute_avg_pixel_values(images, i_masks, o_masks)

    # print(scs.ttest_ind(inner_avg, outer_avg))

    # print(np.mean(inner_avg), '+/-', np.std(inner_avg), inner_avg[:10])
    # print(np.mean(outer_avg), '+/-', np.std(outer_avg), inner_avg[:10])
    # print(np.percentile(inner_avg, [5, 95]))
    # print(np.percentile(outer_avg, [5, 95]))

    # plot_hists(inner_avg, outer_avg, plotname = 'images/hist.png')
    threshold_stats(images, i_masks, o_masks)
