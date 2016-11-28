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
import time
import matplotlib.pyplot as plt
import numpy as np
from mnist_common import prompt_topdir, load_datasets_separate
from mnist_tf_train_gd_sgd import tf_gd_build_graph, tf_gd_train, tf_sgd_build_graph, tf_sgd_train, \
    tf_sgd_build_graph_relu

# Globals
num_labels = 10      # 'A' through 'J'
image_size = 28      # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def run_gd(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    """Run Gradient Descent training and optionally apply L2 regularization"""

    # With gradient descent training, even this much data is prohibitive.
    # Trauncate the training data for faster turnaround.
    train_subset = 20000
    train_dataset_t = train_dataset[:train_subset, :]
    train_labels_t = train_labels[:train_subset]
    print("Building Gradient Descent Graph using %s training labels" % train_subset)
    graph, helpers = tf_gd_build_graph(train_dataset_t, train_labels_t, valid_dataset, test_dataset,
                                       num_labels, image_size)
    num_steps = 2050 #800
    train_labels_t = train_labels[:train_subset, :]
    print("Starting training using Gradient Descent (num_steps=%s)..."  % num_steps)
    start = time.time()
    tf_gd_train(graph, num_steps, helpers, train_labels_t, valid_labels, test_labels)
    end = time.time()
    print("Training completed (elapsed time = %s seconds)." % (end-start))

def display_acc_vs_reg_results(acc_list, reg_list):
    """Plot Accuracy vs regularization"""
    print("L2 regularization 'beta': Test accuracy")
    print("=======================================")
    for (reg, acc) in list(zip(reg_list, acc_list)):
        print("%6f: %.1f%%" % (reg, acc))
    print('')
    plt.semilogx(reg_list, acc_list)
    plt.grid(True)
    plt.title("Test accuracy vs L2 regularization 'beta'")
    plt.show()

def run_sgd(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
            apply_regularization=False):
    """Run Stochastic Gradient Descent training"""
    batch_size = 128
    print("Building Stochastic Gradient Descent Graph using batch size =", batch_size)
    graph, helpers = tf_sgd_build_graph(batch_size, valid_dataset, test_dataset, num_labels, image_size)
    num_steps = 3000 # 10000 if not apply_regularization else 3000
    # You can enduce overfitting by changing num_batches to small # such as 3
    num_batches = 0
    reg_list = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    reg_list = [1e-3] # Just use one value for now
    l2_reg_beta_l = reg_list if apply_regularization else [0.]
    acc_list = []
    verbose = True if not apply_regularization else False
    for l2_reg_beta in l2_reg_beta_l:
        print("Starting training using Stochastic Gradient Descent %s (num_steps=%s)..." %
              ('with L2 regularization beta=%f' % l2_reg_beta if apply_regularization else 'without regularization',
               num_steps))
        start = time.time()
        accuracy = tf_sgd_train(graph, num_steps, batch_size, helpers,
                                train_dataset, train_labels, valid_labels, test_labels,
                                l2_reg_beta=l2_reg_beta, verbose=verbose, num_batches=num_batches)
        end = time.time()
        print("Training completed (elapsed time = %s seconds).\n" % (end - start))
        acc_list.append(accuracy)
    if apply_regularization and len(reg_list) > 1:
        display_acc_vs_reg_results(acc_list, reg_list)

def run_sgd_relu(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                 apply_regularization=False, use_dropout=False):
    """Run Stochastic Gradient Descent training"""
    batch_size = 128
    num_nodes = 1024 # no of nodes in hidden layer
    print("Building Stochastic Gradient Descent Graph using batch size = %s and a %s node RELU hidden layer" %
          (batch_size, num_nodes))
    graph, helpers = tf_sgd_build_graph_relu(batch_size, num_nodes, valid_dataset, test_dataset,
                                             num_labels, image_size, use_dropout=use_dropout)
    num_steps = 3000 #10000 if not apply_regularization else 3000
    # You can enduce overfitting by changing num_batches to small # such as 3
    num_batches = 0
    reg_list = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    reg_list = [1e-3] # Just use one value for now
    l2_reg_beta_l = reg_list if apply_regularization else [0.]
    acc_list = []
    verbose = True if not apply_regularization else False
    for l2_reg_beta in l2_reg_beta_l:
        print("Starting training using Stochastic Gradient Descent %s %s(num_steps=%s) ..." %
              ('with L2 regularization beta=%f' % l2_reg_beta if apply_regularization else 'without regularization',
               'using dropout ' if use_dropout else '', num_steps))
        start = time.time()
        accuracy = tf_sgd_train(graph, num_steps, batch_size, helpers,
                                train_dataset, train_labels, valid_labels, test_labels,
                                l2_reg_beta=l2_reg_beta, verbose=verbose, num_batches=num_batches)
        end = time.time()
        print("Training completed (elapsed time = %s seconds).\n" % (end - start))
        acc_list.append(accuracy)
    if apply_regularization and len(reg_list) > 1:
        display_acc_vs_reg_results(acc_list, reg_list)

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

    # 1st run gradient descent
    run_gd(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    # switch to stochastic gradient descent training instead, which is much faster
    run_sgd(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    # Now run with regularization
    run_sgd(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
            apply_regularization=True)
    # Now stochastic gradient descent training with RELU hidden layer, which should be more accurate
    run_sgd_relu(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    # Now run with regularization
    run_sgd_relu(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                 apply_regularization=True)
    # Redo with dropout
    run_sgd_relu(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                 use_dropout=True)

    #print('label', train_labels[1], ':\n', train_dataset[1])
    # san_file = reg_file.replace('.pickle', '_sanitized.pickle')
    # success, train_dataset_s, train_labels_s, valid_dataset_s, valid_labels_s, test_dataset_s, test_labels_s = \
    #     load_datasets_separate(san_file, reshape=True, num_labels=num_labels, image_size=image_size,
    #                            verbose=True, description='Sanitized')
    # if not success:
    #     print("...Skipping it!")  # continue anyway
    #print('label', train_labels_s[1], ':\n', train_dataset_s[1])
    return True

if __name__ == '__main__':
    rc = main()
    sys.exit(0 if rc else 1)
