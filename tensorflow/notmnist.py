#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


url = 'http://yaroslavvb.com/upload/notMNIST/'
num_classes = 10
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
np.random.seed(133)
pickle_file = 'notMNIST.pickle'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print 'Found and verified', filename
    else:
        raise Exception(
            'Failed to verify:' + url + filename + '. Can you get to it with a browser?')
    return filename


def extract(filename):
    tar = tarfile.open(filename)
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.exists(root):
        print('Already extracted for %s' % root)
        return [os.path.join(root, dir) for dir in os.listdir(root)]
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if d != '.DS_Store'
    ]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load(data_folders, min_num_images, max_num_images):
    dataset = np.ndarray(
        shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0
    image_index = 0

    for folder in data_folders:
        print(folder)
        for image in os.listdir(folder):
            if image_index >= max_num_images:
                raise Exception('More images than expected: %d >= %d' % (
                    image_index, max_num_images))
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) -
                              pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e,
                      '- it\'s ok, skipping.')
        label_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (
            num_images, min_num_images))
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    print('Labels:', labels.shape)
    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def flatten_dataset(dataset):
    dataset_flat = [data.flatten() for data in dataset]
    return np.array(dataset_flat)


def main():
    if not os.path.exists(pickle_file):
        train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
        test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

        train_folders = extract(train_filename)
        test_folders = extract(test_filename)

        train_dataset, train_labels = load(train_folders, 450000, 550000)
        test_dataset, test_labels = load(test_folders, 18000, 20000)

        train_dataset, train_labels = randomize(train_dataset, train_labels)
        test_dataset, test_labels = randomize(test_dataset, test_labels)

        train_size = 200000
        valid_size = 10000

        valid_dataset = train_dataset[:valid_size, :, :]
        valid_labels = train_labels[:valid_size]
        train_dataset = train_dataset[valid_size:valid_size+train_size, :, :]
        train_labels = train_labels[valid_size:valid_size+train_size]

        try:
            f = open(pickle_file, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
    else:
        # load pickle file
        print 'Load from saved data:', pickle_file
        f = open(pickle_file, 'rb')
        data = pickle.load(f)

        # assign each variable
        train_dataset = data['train_dataset']
        train_labels = data['train_labels']
        test_dataset = data['test_dataset']
        test_labels = data['test_labels']
        valid_dataset = data['valid_dataset']
        valid_labels = data['valid_labels']

    # print details of the notMNIST dataset
    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # flatten dataset for using sklearn
    train_dataset_flat = flatten_dataset(train_dataset)
    test_dataset_flat = flatten_dataset(test_dataset)
    valid_dataset_flat = flatten_dataset(valid_dataset)

    # problem6
    clf = LogisticRegression()
    clf.fit(train_dataset_flat[:5000], train_labels[:5000])
    pred = clf.predict(test_dataset_flat[:10000])
    labels = map(np.str, np.unique(test_labels[:10000]))
    print classification_report(test_labels[:10000], pred, target_names=labels)

if __name__ == '__main__':
    main()
