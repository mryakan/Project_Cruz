#!/usr/bin/python3

#============================================================================
# Name        : mnist_train_sgd.py
# Author      : Medhat R. Yakan
# Version     : 1.0
#
# ****************************************************************************
# Copyright   : Copyright (c) 2016 "Medhat R. Yakan" - All Rights Reserved
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
# ****************************************************************************
#
# Description : See docstring
#
#============================================================================

"""
Train a logistic regression classifier on the mnist data using Stochastic Gradient Descent (SGD)
to solve assignment stated here:
 'https://classroom.udacity.com/courses/ud730/lessons/6379031992/concepts/65959889480923#'
"""
import os
import sys
from mnist_common import prompt_topdir, load_datasets_separate

# Globals
num_labels = 10      # 'A' through 'J'
image_size = 28      # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def main():
    """main fn"""
    # Prompt for top directory where mnist folders are located
    topdir = prompt_topdir("pickled dataset files")
    if not topdir:
        return None, None

    # Pickled datasets, both original and sanitized
    dataset_filename = 'notMNIST.pickle'
    reg_file = os.path.join(topdir, dataset_filename)

    # Let's load the datasets & Reshape the data
    success, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_datasets_separate(reg_file, reshape=True, num_labels=num_labels, image_size=image_size,
                               verbose=True, description='Regular')
    if not success:
        print("...Aborting.")
        return False

    #print('label', train_labels[1], ':\n', train_dataset[1])
    san_file = reg_file.replace('.pickle', '_sanitized.pickle')
    success, train_dataset_s, train_labels_s, valid_dataset_s, valid_labels_s, test_dataset_s, test_labels_s = \
        load_datasets_separate(san_file, reshape=True, num_labels=num_labels, image_size=image_size,
                               verbose=True, description='Sanitized')
    if not success:
        print("...Skipping it!")  # continue anyway
    #print('label', train_labels_s[1], ':\n', train_dataset_s[1])
    return True

if __name__ == '__main__':
    rc = main()
    sys.exit(0 if rc else 1)
