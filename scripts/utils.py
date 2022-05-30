# Define some utility functions to load and read the .tif images from the files
# listing subdirectories
# loop through the files in LiveEO_ML_intern_challenge folder  
# and combine, into a dictionary, the image and its corresponding mask


import os
import glob

def combine_files(images, labels, idx):
    files = {'image': images[idx], 
             'mask': labels[idx]}
    return files

def get_files(base_dir):
    images = sorted(glob.glob(os.path.join(base_dir, r"images/*")))
    labels = sorted(glob.glob(os.path.join(base_dir, r"labels/*")))
    files = [combine_files(images, labels,idx) for idx in range(len(images))]
    return files


