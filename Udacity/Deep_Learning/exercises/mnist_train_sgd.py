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
from tf_train_sgd_lib import tf_sgd_build_graph_reluN, tf_sgd_train
from mnist_common import prompt_topdir, load_datasets_separate
from mnist_tf_train_gd_sgd import tf_gd_build_graph, tf_gd_train, tf_sgd_build_graph, tf_sgd_build_graph_relu

# Globals
num_labels_G = 10      # 'A' through 'J'
image_size_G = 28      # Pixel width and height.
pixel_depth_G = 255.0  # Number of levels per pixel.

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
    return True

def run_gd(datasets):
    """Run Gradient Descent training and optionally apply L2 regularization"""

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = datasets

    # With gradient descent training, even this much data is prohibitive.
    # Trauncate the training data for faster turnaround.
    train_subset = 20000
    train_dataset_t = train_dataset[:train_subset, :]
    train_labels_t = train_labels[:train_subset]
    print("Building Gradient Descent Graph using %s training labels" % train_subset)
    graph, helpers = tf_gd_build_graph(train_dataset_t, train_labels_t, valid_dataset, test_dataset,
                                       num_labels_G, image_size_G)
    num_steps = 2050 #800
    train_labels_t = train_labels[:train_subset, :]
    label_tuple = train_labels_t, valid_labels, test_labels

    print("Starting training using Gradient Descent (num_steps=%s)..."  % num_steps)
    start = time.time()
    tf_gd_train(graph, num_steps, helpers, label_tuple)
    end = time.time()
    print("Training completed (elapsed time = %s seconds)." % (end-start))
    return True

def run_sgd(datasets, apply_regularization=False, verbose=True):
    """Run Stochastic Gradient Descent training"""

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = datasets # pylint: disable=W0612

    batch_size = 128

    print("Building Stochastic Gradient Descent Graph using batch size =", batch_size)
    graph, helpers = tf_sgd_build_graph(batch_size, valid_dataset, test_dataset, num_labels_G, image_size_G)

    num_steps = 3000 # 10000 if not apply_regularization else 3000

    # You can induce overfitting by changing num_batches to small # such as 3
    num_batches = 0

    reg_list = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    reg_list = [1e-3] # Just use one value for now

    l2_reg_beta_l = reg_list if apply_regularization else [0.]
    acc_list = []

    for l2_reg_beta in l2_reg_beta_l:
        print("Starting training using Stochastic Gradient Descent %s (num_steps=%s)..." %
              ('with L2 regularization beta=%f' % l2_reg_beta if apply_regularization else 'without regularization',
               num_steps))
        start = time.time()
        accuracy = tf_sgd_train(graph, num_steps, batch_size, helpers, datasets,
                                l2_reg_beta=l2_reg_beta, verbose=verbose, num_batches=num_batches)
        end = time.time()
        print("Training completed (elapsed time = %s seconds).\n" % (end - start))
        acc_list.append(accuracy)

    if apply_regularization and len(reg_list) > 1:
        display_acc_vs_reg_results(acc_list, reg_list)

    return True

def run_sgd_relu(datasets, apply_regularization=False, use_dropout=False, use_decay=False, verbose=True):
    """Run Stochastic Gradient Descent training with one hidden RELU layer"""

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = datasets # pylint: disable=W0612

    batch_size = 128

    num_nodes = 1024 # no of nodes in hidden layer

    print("Building Stochastic Gradient Descent Graph using batch size = %s and a %s node RELU hidden layer" %
          (batch_size, num_nodes))
    graph, helpers = tf_sgd_build_graph_relu(batch_size, num_nodes, valid_dataset, test_dataset,
                                             num_labels_G, image_size_G,
                                             use_dropout=use_dropout, use_exp_decay=use_decay)

    num_steps = 3000 #10000 if not apply_regularization else 3000

    datasize = len(train_labels)
    if num_steps > datasize:
        num_steps = datasize

    # You can induce overfitting by changing num_batches to small # such as 3
    num_batches = 0

    reg_list = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    reg_list = [1e-3] # Just use one value for now

    l2_reg_beta_l = reg_list if apply_regularization else [0.]
    acc_list = []
    for l2_reg_beta in l2_reg_beta_l:
        print("Starting training using Stochastic Gradient Descent %s %s%s(num_steps=%s) ..." %
              ('with L2 regularization beta=%f' % l2_reg_beta if apply_regularization else 'without regularization',
               'using dropout ' if use_dropout else '', 'and exponential decay ' if use_decay else '',
               num_steps))
        start = time.time()
        accuracy = tf_sgd_train(graph, num_steps, batch_size, helpers, datasets,
                                l2_reg_beta=l2_reg_beta, verbose=verbose, num_batches=num_batches)
        end = time.time()
        print("Training completed (elapsed time = %s seconds).\n" % (end - start))
        acc_list.append(accuracy)

    if apply_regularization and len(reg_list) > 1:
        display_acc_vs_reg_results(acc_list, reg_list)

    return True

def run_sgd_reluN(num_nodes_l, datasets, apply_regularization=False, use_dropout=False, use_decay=False, verbose=False):
    """Run Stochastic Gradient Descent training with N hidden RELU units defined by 'num_nodes_l' list"""

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = datasets # pylint: disable=W0612

    batch_size = 128

    print("Building Stochastic Gradient Descent Graph using batch size = %s "
          "and a %s RELU hidden layers (num_nodes=%s)" % (batch_size, len(num_nodes_l), num_nodes_l))
    graph, helpers = tf_sgd_build_graph_reluN(batch_size, num_nodes_l, valid_dataset, test_dataset,
                                              image_size_G*image_size_G, num_labels_G,
                                              use_dropout=use_dropout, use_exp_decay=use_decay)

    num_steps = 3000 #10000 if not apply_regularization else 3000

    datasize = len(train_labels)
    if num_steps > datasize:
        num_steps = datasize

    # You can induce overfitting by changing num_batches to small # such as 3
    num_batches = 0

    reg_list = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    reg_list = [1e-3] # Just use one value for now

    l2_reg_beta_l = reg_list if apply_regularization else [0.]
    acc_list = []
    for l2_reg_beta in l2_reg_beta_l:
        print("Starting training using Stochastic Gradient Descent %s %s%s(num_steps=%s) ..." %
              ('with L2 regularization beta=%f' % l2_reg_beta if apply_regularization else 'without regularization',
               'using dropout ' if use_dropout else '', 'and exponential decay ' if use_decay else '',
               num_steps))
        start = time.time()
        accuracy = tf_sgd_train(graph, num_steps, batch_size, helpers, datasets,
                                l2_reg_beta=l2_reg_beta, verbose=verbose, num_batches=num_batches)
        end = time.time()
        print("Training completed (elapsed time = %s seconds).\n" % (end - start))
        acc_list.append(accuracy)

    if apply_regularization and len(reg_list) > 1:
        display_acc_vs_reg_results(acc_list, reg_list)

    return True

def run_training(): # pylint: disable=R0912
    """Run various training algorithms on datasets"""

    # Prompt for top directory where mnist folders are located
    topdir = prompt_topdir("pickled dataset files")
    if not topdir:
        return False

    # Pickled datasets, both original and sanitized
    dataset_filename = 'notMNIST.pickle'
    reg_file = os.path.join(topdir, dataset_filename)

    # Let's load the datasets & Reshape the data
    success, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_datasets_separate(reg_file, reshape=True, num_labels=num_labels_G, image_size=image_size_G,
                               verbose=True, description='Regular')
    if not success:
        print("...Aborting.")
        return False

    datasets = (train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

    # Switches to enable/disable different subtests:
    en_matrix = {
        'gd': {'Main': False},
        'sgd': {'Main': True,
                'plain': False,
                'plain_reg': False,
                'relu': False,
                'relu_reg': False,
                'relu_reg_alt': False,
                'relu_drop': False,
                'relu_reg_decay': False,
                'relu_reg_3l': False
               },
        'sanitized_data': {'Main': False,
                           'sgd_relu_1_reg': False
                          }
    }

    # 1st run gradient descent
    if en_matrix['gd']['Main']:
        run_gd(datasets)

    if en_matrix['sgd']['Main']:
        # switch to stochastic gradient descent training instead, which is much faster
        if en_matrix['sgd']['plain']:
            run_sgd(datasets)
        # Now run with regularization
        if en_matrix['sgd']['plain_reg']:
            run_sgd(datasets, apply_regularization=True, verbose=False)
        # Now stochastic gradient descent training with RELU hidden layer, which should be more accurate
        if en_matrix['sgd']['relu']:
            run_sgd_relu(datasets)
        # Now run with regularization
        if en_matrix['sgd']['relu_reg']:
            run_sgd_relu(datasets, apply_regularization=True)
        # Alternate identical way
        if en_matrix['sgd']['relu_reg_alt']:
            num_nodes = [1024]  # no of nodes in hidden layers
            run_sgd_reluN(num_nodes, datasets, apply_regularization=True, verbose=True)
        # Redo with dropout
        if en_matrix['sgd']['relu_drop']:
            run_sgd_relu(datasets, use_dropout=True)
        # Now redo with learning rate decay
        if en_matrix['sgd']['relu_reg_decay']:
            run_sgd_relu(datasets, apply_regularization=True, use_decay=True)
        # with 3 layers!
        if en_matrix['sgd']['relu_reg_3l']:
            num_nodes = [1024, 512, 256]  # no of nodes in hidden layers
            run_sgd_reluN(num_nodes, datasets, apply_regularization=True, verbose=True)

    # Now Load sanitized data and train on it
    if en_matrix['sanitized_data']['Main']:
        #print('label', train_labels[1], ':\n', train_dataset[1])
        san_file = reg_file.replace('.pickle', '_sanitized.pickle')
        success, train_dataset_s, train_labels_s, valid_dataset_s, valid_labels_s, test_dataset_s, test_labels_s = \
            load_datasets_separate(san_file, reshape=True, num_labels=num_labels_G, image_size=image_size_G,
                                   verbose=True, description='Sanitized')
        datasets_s = train_dataset_s, train_labels_s, valid_dataset_s, valid_labels_s, test_dataset_s, test_labels_s
        if not success:
            print("...Skipping it!")  # continue anyway
            return True
        #print('label', train_labels_s[1], ':\n', train_dataset_s[1])
        if en_matrix['sanitized_data']['sgd_relu_1_reg']:
            run_sgd_relu(datasets_s, apply_regularization=True, use_decay=True)
    return True

def main():
    """main fn"""
    return run_training()

if __name__ == '__main__':
    rc = main()
    sys.exit(0 if rc else 1)
