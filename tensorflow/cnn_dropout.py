#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import uda_tf_util

# Parameters
num_labels = 10
image_size = 28
batch_size = 16
num_steps = 8001
patch_size = 5

# Network parameters
n_input = image_size * image_size
n_classes = num_labels
learning_rate = 1e-4
n_featmap1 = 32
n_featmap2 = 64
n_fconnected = 1024

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, k, k, 1], padding="SAME")


def conv_net(x, w, b, keep_prob):
    # Reshape
    x = tf.reshape(x, [-1, image_size, image_size, 1])

    # First convolutional layer
    conv1 = tf.nn.relu(conv2d(x, w['wc1']) + b['bc1'])
    pool1 = max_pool(conv1, k=2)

    # Second convolutional layer
    conv2 = tf.nn.relu(conv2d(pool1, w['wc2']) + b['bc2'])
    pool2 = max_pool(conv2, k=2)

    # Densely connected layer and dropout
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * n_featmap2])
    fc = tf.nn.relu(tf.matmul(pool2_flat, w['wf']) + b['bf'])
    fc_drop = tf.nn.dropout(fc, keep_prob)

    # Readout layer
    y_conv = tf.nn.softmax(tf.matmul(fc_drop, w['wo']) + b['bo'])
    return y_conv


def main():
    # load the notMNIST data
    train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels = uda_tf_util.load_pickle('notMNIST.pickle')

    # reformat
    train_dataset, train_labels = uda_tf_util.reformat(train_dataset,
                                                       train_labels, num_labels,
                                                       is_flat=False)
    valid_dataset, valid_labels = uda_tf_util.reformat(valid_dataset,
                                                       valid_labels, num_labels,
                                                       is_flat=False)
    test_dataset, test_labels = uda_tf_util.reformat(test_dataset,
                                                     test_labels, num_labels,
                                                     is_flat=False)
    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # Tensor graph
    graph = tf.Graph()
    with graph.as_default():
        # Input dataset
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(None, image_size, image_size, 1))
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        weights = {
            'wc1': tf.Variable(tf.truncated_normal([patch_size, patch_size,
                                                    1, n_featmap1],
                                                   stddev=0.1)),
            'wc2': tf.Variable(tf.truncated_normal([patch_size, patch_size,
                                                    n_featmap1, n_featmap2],
                                                   stddev=0.1)),
            'wf': tf.Variable(
                tf.truncated_normal([image_size/4 * image_size/4 * n_featmap2,
                                     n_fconnected], stddev=0.1)),
            'wo': tf.Variable(tf.truncated_normal([n_fconnected, n_classes],
                                                  stddev=0.1))
        }
        biases = {
            'bc1': tf.Variable(tf.zeros([n_featmap1])),
            'bc2': tf.Variable(tf.zeros([n_featmap2])),
            'bf': tf.Variable(tf.zeros([n_fconnected])),
            'bo': tf.Variable(tf.zeros([n_classes]))
        }
        keep_prob = tf.placeholder(tf.float32)

        pred = conv_net(tf_train_dataset, weights, biases, keep_prob)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(pred, tf_train_labels)
        )
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        train_prediction = pred
        valid_prediction = conv_net(tf_valid_dataset, weights, biases, 1.0)
        test_prediction = conv_net(tf_test_dataset, weights, biases, 1.0)

    # Graph computation
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'Initialized.'

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels, keep_prob: 0.5}

            # Run a training
            _, l, predictions = session.run([optimizer,
                                             loss,
                                             train_prediction],
                                            feed_dict=feed_dict)
            
            # Print results
            if step % 50 == 0:
                print 'Minibatch loss at step %d: %f' % (step, l)
                print 'Minibatch accuracy: %.5f%%' % accuracy(predictions,
                                                              batch_labels)
                print 'Validation accuracy: %.5f%%' % accuracy(
                    valid_prediction.eval(), valid_labels)
        print 'Test accuracy: %.5f%%' % accuracy(test_prediction.eval(),
                                                 test_labels)

if __name__ == '__main__':
    main()
