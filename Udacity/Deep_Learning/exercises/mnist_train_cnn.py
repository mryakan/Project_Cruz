#!/usr/bin/python3

#============================================================================
# Name        : mnist_train_cnn.py
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
Train a Convolutional Neural Network on the mnist data to solve assignment stated here:
 'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb'
"""
import os
import sys
import time
import tensorflow as tf
from mnist_common import prompt_topdir, load_datasets_separate
from tf_train_sgd_lib import init_weight_var, init_bias_var, tf_sgd_train

# Globals
num_labels_G = 10      # 'A' through 'J'
image_size_G = 28      # Pixel width and height.
pixel_depth_G = 255.0  # Number of levels per pixel.
num_channles_G = 1     # ==> Grayscale

def init_weights_and_biases_cnn(graph, patch_size, depth, num_hidden, image_size, num_labels, num_channels):
    """Initialize the weights and biases for a Convolutional graph"""
    with graph.as_default():
        weights_l = []
        biases_l = []
        stddev = 0.1
        # 1st layer
        weights_l.append(init_weight_var([patch_size, patch_size, num_channels, depth], stddev=stddev))
        biases_l.append(init_bias_var(0, [depth]))
        # 2nd layer
        weights_l.append(init_weight_var([patch_size, patch_size, depth, depth], stddev=0.1))
        biases_l.append(init_bias_var(1.0, [depth]))
        # 3rd layer
        weights_l.append(init_weight_var([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        biases_l.append(init_bias_var(1.0, [num_hidden]))
        # 4th layer
        weights_l.append(init_weight_var([num_hidden, num_labels], stddev=0.1))
        biases_l.append(init_bias_var(1.0, [num_labels]))

    return weights_l, biases_l

def forward_prop_cnn(graph, data, weights_l, biases_l):
    """Convolutional Neural network forward propagation"""
    with graph.as_default():
        conv = tf.nn.conv2d(data, weights_l[0], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + biases_l[0])
        conv = tf.nn.conv2d(hidden, weights_l[1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + biases_l[1])
        shape = hidden.get_shape().as_list() # pylint: disable=E1101
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, weights_l[2]) + biases_l[2])
    return tf.matmul(hidden, weights_l[3]) + biases_l[3]

def tf_build_graph_cnn(batch_size, patch_size, depth, num_hidden, valid_dataset, test_dataset,  # pylint: disable=R0913
                       image_size, num_labels, num_channels):
    """
    Build model using a small network with two convolutional layers,
    followed by one fully connected layer.
    Convolutional networks are more expensive computationally,
    so limit its depth and number of fully connected nodes
    """
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        #   For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        tf_l2_reg_beta = tf.placeholder(tf.float32) # ignore for now

        # Variables.
        weights_l, biases_l = init_weights_and_biases_cnn(graph, patch_size, depth, num_hidden,
                                                          image_size, num_labels, num_channels)


        # Training computation.
        logits = forward_prop_cnn(graph, tf_train_dataset, weights_l, biases_l)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(forward_prop_cnn(graph, tf_valid_dataset, weights_l, biases_l))
        test_prediction = tf.nn.softmax(forward_prop_cnn(graph, tf_test_dataset, weights_l, biases_l))

    helpers = (optimizer, loss, train_prediction, valid_prediction, test_prediction,
               tf_train_dataset, tf_train_labels, tf_l2_reg_beta)
    return graph, helpers


def run_cnn(datasets):
    """
    Run Convolutional Neural Network model
    """

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = datasets # pylint: disable=W0612

    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64

    print("Building Graph with 2 layer Convolutional Neural Network followed by one fully connected layer "
          "using batch size = %s, patch_size = %s, depth = %s, num_hidden=%s " %
          (batch_size, patch_size, depth, num_hidden))

    graph, helpers = tf_build_graph_cnn(batch_size, patch_size, depth, num_hidden,
                                        valid_dataset, test_dataset,
                                        image_size_G, num_labels_G, num_channles_G)

    num_steps = 10000

    datasize = len(train_labels)
    if num_steps > datasize:
        num_steps = datasize

    # You can induce overfitting by changing num_batches to small # such as 3
    num_batches = 0

    start = time.time()
    tf_sgd_train(graph, num_steps, batch_size, helpers, datasets,
                 verbose=True, num_batches=num_batches)
    end = time.time()
    print("Training completed (elapsed time = %s seconds).\n" % (end - start))

    return True

def run_training():
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
        load_datasets_separate(reg_file, reshape=True, conv_num_chan=num_channles_G,
                               num_labels=num_labels_G, image_size=image_size_G,
                               verbose=True, description='Regular')
    if not success:
        print("...Aborting.")
        return False

    datasets = (train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

    # Switches to enable/disable different subtests:
    en_matrix = {
        'orig_data': {'Main': True,
                      'cnn_basic': True
                     },
        'sanitized_data': {'Main': True,
                           'cnn_basic': True
                          }
    }

    # 1st run foo
    if en_matrix['orig_data']['Main']:
        if en_matrix['orig_data']['cnn_basic']:
            run_cnn(datasets)

    # Now Load sanitized data and train on it
    if en_matrix['sanitized_data']['Main']:
        #print('label', train_labels[1], ':\n', train_dataset[1])
        san_file = reg_file.replace('.pickle', '_sanitized.pickle')
        success, train_dataset_s, train_labels_s, valid_dataset_s, valid_labels_s, test_dataset_s, test_labels_s = \
            load_datasets_separate(san_file, reshape=True, conv_num_chan=num_channles_G,
                                   num_labels=num_labels_G, image_size=image_size_G,
                                   verbose=True, description='Sanitized')
        datasets_s = train_dataset_s, train_labels_s, valid_dataset_s, valid_labels_s, test_dataset_s, test_labels_s
        if not success:
            print("...Skipping it!")  # continue anyway
            return True
        #print('label', train_labels_s[1], ':\n', train_dataset_s[1])
        if en_matrix['sanitized_data']['cnn_basic']:
            run_cnn(datasets_s)
    return True

def main():
    """main fn"""
    return run_training()

if __name__ == '__main__':
    rc = main()
    sys.exit(0 if rc else 1)
