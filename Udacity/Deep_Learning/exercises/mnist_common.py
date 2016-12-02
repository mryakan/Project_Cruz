#============================================================================
# Name        : mnist_common.py
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
Common utlities to be used by the various mnist deelp learning modules.
"""

import os
import pickle
import numpy as np

def prompt_topdir(description="files"):
    """Prompt user for top level directory"""
    topdir = input("Enter top level directory containing the %s [default='.']:\n" % description)
    if not topdir or topdir == '.':
        topdir = os.getcwd()
    ndir = os.path.abspath(topdir)
    if not os.path.exists(ndir) or not os.path.isdir(ndir):
        print("ERROR: Invalid Directory '%s'" % ndir)
        return None
    return topdir


def load_datasets(pickle_file):
    """Load the data"""
    try:
        with open(pickle_file, 'rb') as f:
            datasets = pickle.load(f)
            f.close()
    except IOError as e:
        print('ERROR: Unable to open file', pickle_file, ':', e)
        return None
    except EOFError as e:
        print('ERROR: Unable to read data from', pickle_file, ':', e)
        return None
    except pickle.UnpicklingError as e:
        print('ERROR: Unable to unpickle data from', pickle_file, ':', e)
        return None
    return datasets

def reformat_data(dataset, labels, num_labels, image_size):
    """
    Reformat into a shape that's more adapted to the models we're going to train:
      - data as a flat matrix
      - labels as float 1-hot encodings
    """
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32) # pylint: disable=E1101
    # Map 0 to [1.0, 0.0, 0.0  ], 1 to [0.0, 1.0, 0.0  ]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32) # pylint: disable=E1101
    return dataset, labels

def reformat_data_conv(dataset, labels, num_labels, image_size, num_channels):
    """
    Reformat into a shape that's more adapted to the convolutional models we're going to train:
      - convolutions need the image data formatted as a cube (width by height by #channels)
      - labels as float 1-hot encodings
    """
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32) # pylint: disable=E1101
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32) # pylint: disable=E1101
    return dataset, labels

def load_datasets_separate(pickle_file, reshape=False, conv_num_chan=0, num_labels=0, image_size=0,
                           verbose=True, description=''):
    """Load the data into separate arrays"""
    if verbose:
        print("Trying to load", description, "dataset from pickle file", pickle_file, ' ')
    datasets = load_datasets(pickle_file)
    if not datasets:
        print("ERROR: Unable to load %s dataset." % description)
        return False, None, None, None, None, None, None
    print("Loaded", description, "data.")
    train_dataset = datasets['train_dataset']
    train_labels = datasets['train_labels']
    valid_dataset = datasets['valid_dataset']
    valid_labels = datasets['valid_labels']
    test_dataset = datasets['test_dataset']
    test_labels = datasets['test_labels']
    if verbose:
        print('  Loaded', description, 'Training set: ', train_dataset.shape, train_labels.shape)
        print('  Loaded', description, 'Validation set: ', valid_dataset.shape, valid_labels.shape)
        print('  Loaded', description, 'Test set: ', test_dataset.shape, test_labels.shape)
    if reshape:
        if not num_labels or not image_size:
            print("ERROR: num_bales and imagesize must be > 0.")
            return False, None, None, None, None, None, None
        if verbose:
            print("Reformatting", description, "data...")
        if conv_num_chan:
            train_dataset, train_labels = reformat_data_conv(train_dataset, train_labels, num_labels, image_size,
                                                             conv_num_chan)
            valid_dataset, valid_labels = reformat_data_conv(valid_dataset, valid_labels, num_labels, image_size,
                                                             conv_num_chan)
            test_dataset, test_labels = reformat_data_conv(test_dataset, test_labels, num_labels, image_size,
                                                           conv_num_chan)
        else:
            train_dataset, train_labels = reformat_data(train_dataset, train_labels, num_labels, image_size)
            valid_dataset, valid_labels = reformat_data(valid_dataset, valid_labels, num_labels, image_size)
            test_dataset, test_labels = reformat_data(test_dataset, test_labels, num_labels, image_size)
        if verbose:
            print('  Reshaped', description, 'Training set:', train_dataset.shape, train_labels.shape)
            print('  Reshaped', description, 'Validation set:', valid_dataset.shape, valid_labels.shape)
            print('  Reshaped', description, 'Test set:', test_dataset.shape, test_labels.shape)
    return True, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def calc_accuracy(predictions, labels):
    """Calculate accuracy of predictions"""
    acc = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    return acc
