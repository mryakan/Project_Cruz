#!/usr/bin/python3

#============================================================================
# Name        : softmax.py
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
A softmax implementation
"""

import sys
from time import time
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce



def my_softmax1(x):
    """Compute the softmax values for each set of scores in the numpy array x"""
    if not isinstance(x, np.ndarray):
        x = np.array(x, ndmin=2).T
    y = np.array([])
    for row in x.T:
        srow = list(exp(elem) for elem in row)
        sumexp = sum(srow)
        srow = list(exp(elem)/sumexp for elem in row)
        y = np.array(srow) if not y.any() else np.vstack((y, srow)) # pylint: disable=E1101
    return y.T  # pylint: disable=E1101

def my_softmax2(x):
    """Compute the softmax values for each set of scores in the numpy array x"""
    if not isinstance(x, np.ndarray):
        x = np.array(x, ndmin=2).T
    y = np.array([])
    for i in range(x.shape[1]):
        col = x[:, i]
        sumexp = reduce(lambda a, b: a + exp(b), col, 0)
        scol = list(map(lambda x: exp(x)/sumexp, col))
        y = np.array(scol) if not y.any() else np.vstack((y, scol)) # pylint: disable=E1101
    return y.T  # pylint: disable=E1101

def softmax_np(x):
    """Compute the softmax values for each set of scores in the numpy array x using numpy libs"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax(x, timeit=True):
    """Compute the softmax values for each set of scores in the numpy array x"""
    fns = [my_softmax1, my_softmax2, softmax_np]
    for fn in fns:
        if timeit:
            print("Usinf fn %s():" % fn.__name__)
            stime = time()
        result =  fn(x)
        if timeit:
            etime = time() - stime
            print("Elapsed time = %s seconds." % etime)
    return result


def test():
    """Test softmax fn"""
    scores = [3.0, 1.0, 0.2]
    print("scores: \n%s" % scores)
    print("softmax(scores): \n%s" % softmax(scores))
    # Answer should be [ 0.8360188   0.11314284  0.05083836]

    scores = [1.0, 2.0, 3.0]
    print("scores: \n%s" % scores)
    print("softmax(scores): \n%s" % softmax(scores))
    # Answer should be [ 0.09003057  0.24472847  0.66524096]

    scores = np.array([[1, 2, 3, 6],
                       [2, 4, 5, 6],
                       [3, 8, 7, 6]])
    print("scores: \n%s" % scores)
    print("softmax(scores): \n%s" % softmax(scores))
    # It should return a 2-dimensional array of the same shape, (3, 4):
    # [[ 0.09003057  0.00242826  0.01587624  0.33333333]
    #  [ 0.24472847  0.01794253  0.11731043  0.33333333]
    #  [ 0.66524096  0.97962921  0.86681333  0.33333333]]


def plot():
    """Plot softmax curves"""
    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
    #print("scores: %s" % scores)
    #print("softmax(scores): %s" % softmax(scores))

    plt.plot(x, softmax(scores).T, linewidth=2) # pylint: disable=E1101
    plt.show()


def main():
    """main fn"""
    test()
    plot()
    return True

if __name__ == '__main__':
    rc = main()
    sys.exit(0 if rc else 1)
