#!/usr/bin/python3

#============================================================================
# Name        : mnist_train_logreg.py
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
Train a logistic regression classifier on the mnist data to solve Problem 6 stated here:
 'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb'
"""


import os
import sys
import pickle
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score

def create_logreg(cross_validate=True):
    """Create a logistic regression classifier with or without Cross Validation"""
    # For now, use defaults!
    if cross_validate:
        logreg = LogisticRegressionCV()
    else:
        logreg = LogisticRegression(verbose=0) # pylint: disable=R0204
    return logreg

def train_logreg(datasets, datasizes, use_cv):
    """
    Train the logistic regression classifier using specified dataset sizes from the training set in the given dataset
    then check prediction scores
    return list containing results and corresponding scores per datatset.
    """
    results = []
    scores = []
    maxsize = len(datasets['train_dataset'])
    logreg = create_logreg(cross_validate=use_cv)
    if not logreg:
        print("ERROR: Unable to create Logistic Regression Classifier instance. Aborting!")
        return None
    for datasize in datasizes:
        if datasize > maxsize:
            print("ERROR: Specified datasize is too large (%s > %s)" % (datasize, maxsize))
            return None
        if datasize == 0:
            datasize = maxsize
        # Select and Flatten data
        t_data = [d.flatten() for d in datasets['train_dataset'][:datasize]]
        t_labels = datasets['train_labels'][:datasize]
        print("Fitting using training data (dataset size=%s)..." % datasize)
        result = logreg .fit(t_data, t_labels)
        results.append(result)
        print("Calculating cross validation scores on training data...")
        score = cross_val_score(result, t_data, t_labels)
        scores.append(score)
    return result, scores

def load_dataset(pickle_file):
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

def main():
    """main fn"""
    # Prompt for top directory where mnist folders are located
    topdir = input("Enter top level directory containing the pickled dataset files [default='.']:\n")
    if not topdir or topdir == '.':
        topdir = os.getcwd()
    ndir = os.path.abspath(topdir)
    if not os.path.exists(ndir) or not os.path.isdir(ndir):
        print("ERROR: Invalid Directory '%s'" % ndir)
        return None, None

    # Pickled datasets, both original and sanitized
    dataset_filename = 'notMNIST.pickle'
    reg_file = os.path.join(topdir, dataset_filename)
    san_file = reg_file.replace('.pickle', '_sanitized.pickle')
    # Let's load the datasets
    print("Trying to load regular dataset from pickle file", reg_file, '...')
    reg_data = load_dataset(reg_file)
    if not reg_data:
        print("ERROR: Unable to load regular dataset. Aborting!")
        return False
    print("Loaded regular data.")
    #print('label', reg_data['train_labels'][1], ':\n', reg_data['train_dataset'][1])
    print("Trying to load sanitized dataset from pickle file", reg_file, '...')
    san_data = load_dataset(san_file)
    print("Trying to load sanitized dataset from pickle file", reg_file, '...')
    if not san_data:
        print("ERROR: Unable to load sanitized dataset. Skipping it!")
    else:
        print("Loaded Sanitized data.")
        #print('label', san_data['train_labels'][1], ':\n', san_data['train_dataset'][1])
    datasizes = [50, 100, 1000, 5000]
    try_all = False
    if try_all:
        datasizes.append(0) # (0 ==> all: May take a LONG time to complete)
    # Train regular data with specified dataset sizes
    for datasel in [(reg_data, "original data"), (san_data, "sanitized data")]:
        for cv in [False, True]:
            print("Training Logistic Regression classifier on %s (CV=%s)..." % (datasel[1], cv))
            _, scores = train_logreg(datasel[0], datasizes, cv)
            for i, datasize in enumerate(datasizes):
                print("Mean accuracy for %s using %s samples: %s" % (datasel[1], datasize, scores[i].mean()))
    return True

if __name__ == '__main__':
    rc = main()
    sys.exit(0 if rc else 1)
