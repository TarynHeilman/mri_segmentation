# MRI SEGMENTATION

## Pre-Processing Pipeline

### Part 1

I experimented with the existing source code, located in [src/parsing.py](https://github.com/TarynHeilman/mri_segmentation/blob/master/src/parsing.py). I made minimal changes to get the code to run - needed to change all references of `dicom` to `pydicom` as the current version requires. Once these changes were made, the functions ran as written.

To verify that contours were being parsed correctly, I wrote a unittest to check the parsing function using a simple square polygon. See the first test in [test/unittests.py](https://github.com/TarynHeilman/mri_segmentation/blob/master/test/unittests.py) for more details.

I also visually inspected a few examples from the dataset, which looked accurate (to my admittedly limited knowledge on the subject!) See example below.

![](images/test_image1.png)

For ease of pre-processing, I created a pipeline class ([src/pipeline.py](https://github.com/TarynHeilman/mri_segmentation/blob/master/src/pipeline.py)). For the first pass, I created the methods `process_one_mask` and `create_masks`. The first reads in image and contour files for a single datapoint, and writes the corresponding boolean mask to disk as a `.npy` file with the same relative file path under the `data/masks` directory. The latter method loops through all of the available image files, checks that the corresponding contour file exists, and if so, calls the first method.

### Part 2
For the next pass, I added methods to the class that allowed for reading in files in small batches for model training. To streamline this process, I slightly modified the methods detailed in part one to include
storing lists of paths to images and the targets masks on the class so that those lists did not need to get re-created at run-time. A very basic unittest was constructed to ensure that those lists were the same length, and referenced the same data points.

I constructed a helper method `read_batch_arrays`, which takes a list of files, reads the two file formats and constructs two numpy arrays for the images and the targets. I implemented a basic unittest for this function to check that the arrays have the same shape, which helps validate earlier steps in the pipeline, but by no means is a comprehensive screen against all bugs or edge cases. The `batch_train` method utilizes scikit-learn's `KFold` class to generate shuffled batches of filenames, which it feeds to the helper method.

The code runs, passes *rudimentary* unit tests, and appears to create the appropriate files, but requires more rigorous testing.

### Future Improvements

* Modularize my code further to enable easier testing. Some of my loop functions border on a little long, and that boxed me in so that I struggled to create effective tests.

* More sophisticated logic on writing mask files - would like to not automatically over-write each time the class model is trained - this seems inefficient. Ideally pipeline will be pickled, so logic that only writes those files if indicated (or if they don't exist) seems ideal.

* Incorporate functions from `src/parsing.py` into the pre-processing pipeline for more streamlined code.  
