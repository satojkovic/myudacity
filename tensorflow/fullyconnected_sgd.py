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
    batch_size = 128
    graph_obj = {}
    graph_obj['batch_size'] = batch_size

    graph = tf.Graph()
    with graph.as_default():
        # Input data
        # For the training data, we use a placeholder that will be fed
        # at run time with a training batch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size,
                                                 image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        graph_obj['tf_train_dataset'] = tf_train_dataset
        graph_obj['tf_train_labels'] = tf_train_labels
        graph_obj['train_dataset'] = train_dataset
        graph_obj['train_labels'] = train_labels
        graph_obj['valid_labels'] = valid_labels
        graph_obj['test_labels'] = test_labels

        # Variables
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels])
        )
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        )
        graph_obj['loss'] = loss

        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        graph_obj['optimizer'] = optimizer

        # Predictions for the training, validations, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights) + biases
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
        tf.initialize_all_variables().run()
        print 'Initialized.'
        batch_size = graph_obj['batch_size']

        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % \
                     (graph_obj['train_labels'].shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = graph_obj['train_dataset'][offset:(offset+batch_size), :]
            batch_labels = graph_obj['train_labels'][offset:(offset+batch_size), :]

            # Prepare a dictionary telling the session where to feed
            # the minibatch. The key of the dictionary is the placeholder node
            # of the graph to be fed, and the value is the numpy array
            # to feed of it.
            feed_dict = {graph_obj['tf_train_dataset']: batch_data,
                         graph_obj['tf_train_labels']: batch_labels}
            _, l, predictions = session.run(
                [graph_obj['optimizer'],
                 graph_obj['loss'],
                 graph_obj['train_prediction']],
                feed_dict=feed_dict
            )

            if (step % 500 == 0):
                print 'Minibatch loss at step %d: %f' % (step, l)
                print 'Minibatch accuracy: %.1f%%' % accuracy(predictions,
                                                              batch_labels)
                print 'Validation accuracy: %.1f%%' % accuracy(
                    graph_obj['valid_prediction'].eval(),
                    graph_obj['valid_labels']
                )
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
