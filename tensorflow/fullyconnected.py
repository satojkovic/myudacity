#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import uda_tf_util


def main():
    # load the notMNIST data
    train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels = uda_tf_util.load_pickle('notMNIST.pickle')

    # print details of the notMNIST dataset
    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

if __name__ == '__main__':
    main()
