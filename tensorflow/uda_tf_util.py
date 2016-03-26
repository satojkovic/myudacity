#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import cPickle as pickle
import numpy as np


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
    return train_dataset, train_labels, valid_dataset, \
        valid_labels, test_dataset, test_labels


def reformat(dataset, labels, num_labels):
    # data as a flat matrix
    # labels as float 1-hot encodings
    data_num, flat_size = dataset.shape[0], dataset.shape[1] * dataset.shape[2]
    dataset = dataset.reshape((-1, flat_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
