#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import uda_tf_util

# Paramters
image_size = 28
num_labels = 10
batch_size = 128
display_step = 1
training_epochs = 15
learning_rate = 0.001


def main():
    # load the notMNIST data
    train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels = uda_tf_util.load_pickle('notMNIST.pickle')

    # print details of the notMNIST dataset
    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # reformat
    train_dataset, train_labels = uda_tf_util.reformat(train_dataset,
                                                       train_labels, num_labels)
    valid_dataset, valid_labels = uda_tf_util.reformat(valid_dataset,
                                                       valid_labels, num_labels)
    test_dataset, test_labels = uda_tf_util.reformat(test_dataset,
                                                     test_labels, num_labels)

    print('[Reformat]')
    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)


if __name__ == '__main__':
    main()
