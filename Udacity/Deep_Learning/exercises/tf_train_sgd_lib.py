#============================================================================
# Name        : tf_train_sgd_lib.py
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
Libraries to support training logistic regression classifier using Stochastic Gradient Descent (SGD)
with support for multi-layer neural networks with RELU (REctified Linear Units) hidden layers
"""

import tensorflow as tf
from mnist_common import calc_accuracy

def init_weight_var(shape, stddev=0):
    """
    Initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients.
    For using ReLU neurons, it is also good practice to initialize them with a slightly positive initial
    bias to avoid "dead neurons.
    See: 'https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html#weight-initialization'
    """
    initv = tf.truncated_normal(shape, stddev=stddev) if stddev else tf.truncated_normal(shape)
    return tf.Variable(initv)

def init_bias_var(val, shape):
    """Initialize Bias to specified value, shape"""
    return tf.Variable(tf.constant(val, shape=shape, dtype=tf.float32))

def init_weights_and_biases(graph, num_hidden_layers, num_hidden_nodes_l, in_dim, out_dim):
    """Initialize the weights and biases for a graph with 'num_hidden_layers' hidden layers"""
    with graph.as_default():
        # 1st (hidden) layer is in_dim -> num_hidden_nodes
        # 2nd layer is num_hidden_nodes[0] -> num_hidden_nodes[1]
        # ...
        # Last layer is num_hidden_nodes[num_hidden_layers-1] -> out_dim
        weights_l = []
        biases_l = []
        for layer in range(num_hidden_layers+1):
            #stddev = np.sqrt(2.0 / num_hidden_nodes[layer - 1]) if layer else None
            stddev = 0.1
            if layer == 0:
                weights = init_weight_var([in_dim, num_hidden_nodes_l[layer]])
                biases = init_bias_var(0, [num_hidden_nodes_l[layer]])
            elif layer < num_hidden_layers:
                weights = init_weight_var([num_hidden_nodes_l[layer - 1], num_hidden_nodes_l[layer]], stddev=stddev)
                biases = init_bias_var(0, [num_hidden_nodes_l[layer]])
            else:
                weights = init_weight_var([num_hidden_nodes_l[num_hidden_layers - 1], out_dim], stddev=stddev)
                biases = init_bias_var(0, [out_dim])
            weights_l.append(weights)
            biases_l.append(biases)
    return weights_l, biases_l

def forward_prop(graph, num_hidden_layers, input_dataset, weights_l, biases_l, use_dropout=False):
    """Do forward propagation"""
    with graph.as_default():
        logits_l = []
        for layer in range(num_hidden_layers+1):
            if layer == 0:
                input_l = input_dataset
            else:
                input_l = logits_l[layer - 1]
            # Calc WX+b for each layer
            logits = tf.matmul(input_l, weights_l[layer]) + biases_l[layer]
            # Apply RELU/Dropout only on hidden layers
            if layer == num_hidden_layers-1:
                logits = tf.nn.relu(logits)
                if use_dropout:
                    keep_prob = 0.5  # drop half, keep half
                    logits = tf.nn.dropout(logits, keep_prob)
            logits_l.append(logits)
    return logits_l

def tf_sgd_build_graph_reluN(batch_size, num_hidden_nodes_l, valid_dataset, test_dataset, # pylint: disable=R0914
                             in_dim, out_dim, use_dropout=False, use_exp_decay=False):
    """
    load all the data into TensorFlow and build the computation graph for stochastic gradient descent training
    Add 'num_hidden_layers' hidden RELU layers with corresponding 'num_hidden_nodes'
    """
    if not isinstance(num_hidden_nodes_l, list):
        print("ERROR: 'num_hidden_nodes' MUST be a list.")
        return None, None
    num_hidden_layers = len(num_hidden_nodes_l)
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        #   For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, in_dim))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, out_dim))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        tf_l2_reg_beta = tf.placeholder(tf.float32)

        # Variables.
        weights_l, biases_l = init_weights_and_biases(graph, num_hidden_layers, num_hidden_nodes_l,
                                                      in_dim, out_dim)

        # Training computation for RELU hidden layers
        logits_l = forward_prop(graph, num_hidden_layers, tf_train_dataset, weights_l, biases_l,
                                use_dropout=use_dropout)
        l2_regularization_param = tf_l2_reg_beta * sum([tf.nn.l2_loss(x) for x in weights_l])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_l[-1], tf_train_labels)) + \
               l2_regularization_param

        # Optimizer.
        starter_learning_rate = 0.5
        if use_exp_decay:
            learn_rate = tf.constant(starter_learning_rate, dtype=tf.float32)
            global_step = tf.Variable(0, trainable=False)
            num_decay_steps = 1000
            decay_rate = 0.96
            learn_rate = tf.train.exponential_decay(learn_rate, global_step,
                                                    num_decay_steps, decay_rate, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)

        else:
            optimizer = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_l[-1])

        logits_v = forward_prop(graph, num_hidden_layers, tf_valid_dataset, weights_l, biases_l,
                                use_dropout=False)
        valid_prediction = tf.nn.softmax(logits_v[-1])

        logits_t = forward_prop(graph, num_hidden_layers, tf_test_dataset, weights_l, biases_l,
                                use_dropout=False)
        test_prediction = tf.nn.softmax(logits_t[-1])

    helpers = (optimizer, loss, train_prediction, valid_prediction, test_prediction,
               tf_train_dataset, tf_train_labels, tf_l2_reg_beta)
    return graph, helpers

def tf_sgd_train(graph, num_steps, batch_size, helpers, datasets, l2_reg_beta=0, verbose=True, num_batches=0): # pylint: disable=R0914
    """
    Run the computation and iterate 'num_steps' times
    Use optional L2 regularization if 'l2_reg_beta' is != 0
    """

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = datasets # pylint: disable=W0612

    optimizer, loss, train_prediction, valid_prediction, test_prediction, \
    tf_train_dataset, tf_train_labels, tf_l2_reg_beta = helpers

    acc = 0
    if num_batches:
        print(">Restricting # of batches to only", num_batches)

    with tf.Session(graph=graph) as session:
        init_op = tf.initialize_all_variables()
        session.run(init_op) # pylint: disable=E1101
        if verbose:
            print('@Initialized...')
        for step in range(num_steps+1):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            if num_batches:
                # restrict learning to a specified # of batches
                offset = step % num_batches
            else:
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_l2_reg_beta: l2_reg_beta}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if verbose:
                if step == num_steps or step % 500 == 0:
                    print("@step", step)
                    print('Minibatch Loss: %f' % l)
                    print('Minibatch accuracy: %.1f%%' % calc_accuracy(predictions, batch_labels))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph dependencies.
                    print('Validation accuracy: %.1f%%' % calc_accuracy(valid_prediction.eval(), valid_labels))
        if verbose:
            print("@Done")
        acc = calc_accuracy(test_prediction.eval(), test_labels)
        print('Test accuracy: %.1f%%' % acc)
    return acc
