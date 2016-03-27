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
num_steps = 3001

# Network paramters
n_hidden = 1024
n_input = image_size * image_size
n_classes = num_labels


def multilayer_perceptron(X, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['hidden']), biases['hidden']))
    return tf.add(tf.matmul(layer1, weights['out']), biases['out'])


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

    # Tensor Graph
    graph = tf.Graph()
    x = tf.placeholder(tf.float32, shape=[None, n_input])
    y = tf.placeholder(tf.float32, shape=[None, n_classes])
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Compute graph
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(training_epochs):
            avg_cost = 0
            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                sess.run(optimizer, feed_dict={x:batch_data, y:batch_labels})
                avg_cost += sess.run(cost, feed_dict={x:batch_data, y:batch_labels}) / num_steps
            if epoch % display_step == 0:
                print 'Epoch:', '%04d' % (epoch + 1), 'cost=', '{:9f}'.format(avg_cost)
        print 'Finished'

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print 'accuracy:', accuracy.eval({x:test_dataset, y:test_labels})

if __name__ == '__main__':
    main()
