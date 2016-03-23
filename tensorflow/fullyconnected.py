#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import uda_tf_util


image_size = 28
num_labels = 10


def reformat(dataset, labels):
    # data as a flat matrix
    # labels as float 1-hot encodings
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def tensorflow_graph(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    train_subset = 10000
    graph_obj = {}

    graph = tf.Graph()
    with graph.as_default():
        # Input data
        # Load the training, valid, test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        graph_obj['train_labels_subset'] = train_labels[:train_subset]
        graph_obj['valid_labels'] = valid_labels
        graph_obj['test_labels'] = test_labels

        # Variables
        # These are the parameters that we are going to be training.
        # The weight matrix will be initialized using random valued
        # following (truncated) normal distribution.
        # The biases are get initialized to zero.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels])
        )
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation
        # We multiply the inputs with the weight matrix, and add biases.
        # We compute the softmax cross entropy.
        # We take the average of this cross entropy across
        # all training examples: that's our loss
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        )
        graph_obj['loss'] = loss

        # Optimizer
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        graph_obj['optimizer'] = optimizer

        # Predictions for the training, validations, and test data.
        # These are not part of training, but merely here so that we can
        # report accuracy figures as we train
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(valid_dataset, weights) + biases
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(test_dataset, weights) + biases
        )
        graph_obj['train_prediction'] = train_prediction
        graph_obj['valid_prediction'] = valid_prediction
        graph_obj['test_prediction'] = test_prediction

    # return the graph object
    graph_obj['graph'] = graph
    return graph_obj


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def tensorflow_session(graph_obj):
    num_steps = 8001

    with tf.Session(graph=graph_obj['graph']) as session:
        # This is one time operation which ensures the parameters get initialized
        # as we described in the graph: random weights for the matrix,
        # zeros for the biases.
        tf.initialize_all_variables().run()
        print 'Initialized.'

        for step in range(num_steps):
            # Run the computations
            # We tell run() that we want to run the optimizer, and get
            # the loss value and the training predictions returned as
            # numpy arrays
            _, l, predictions = session.run([graph_obj['optimizer'],
                                             graph_obj['loss'],
                                             graph_obj['train_prediction']])

            if step % 100 == 0:
                print 'Loss at step %d: %f' % (step, l)
                print 'Training accuracy: %.1f%%' % accuracy(predictions,
                                                             graph_obj['train_labels_subset'])
                print 'Validation accuracy: %.1f%%' % accuracy(graph_obj['valid_prediction'].eval(),
                                                               graph_obj['valid_labels'])

        print 'Test accuracy: %.1f%%' % accuracy(graph_obj['test_prediction'].eval(),
                                                 graph_obj['test_labels'])


def main():
    # load the notMNIST data
    train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels = uda_tf_util.load_pickle('notMNIST.pickle')

    # print details of the notMNIST dataset
    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # reformat
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # tensorflow
    graph = tensorflow_graph(train_dataset, train_labels,
                             valid_dataset, valid_labels,
                             test_dataset, test_labels)

    # tensorflow graph computation
    tensorflow_session(graph)

if __name__ == '__main__':
    main()
