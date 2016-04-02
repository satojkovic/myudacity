#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import uda_tf_util
import math

# Parameters
num_labels = 10
image_size = 28
batch_size = 128
lambda_2 = 0.001
training_epochs = 15
learning_rate = 1e-4
num_steps = 140001

# Network parameters
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 256
n_hidden4 = 128
n_input = image_size * image_size
n_classes = num_labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def feedforward(dataset, weights, biases, keep_prob):
    layer1 = tf.nn.relu(tf.add(tf.matmul(dataset, weights['hidden1']),
                               biases['hidden1']))
    dropout_layer1 = tf.nn.dropout(layer1, keep_prob)

    layer2 = tf.nn.relu(tf.add(tf.matmul(dropout_layer1,
                                         weights['hidden2']),
                               biases['hidden2']))
    dropout_layer2 = tf.nn.dropout(layer2, keep_prob)

    layer3 = tf.nn.relu(tf.add(tf.matmul(dropout_layer2,
                                         weights['hidden3']),
                               biases['hidden3']))
    dropout_layer3 = tf.nn.dropout(layer3, keep_prob)

    layer4 = tf.nn.relu(tf.add(tf.matmul(dropout_layer3,
                                         weights['hidden4']),
                               biases['hidden4']))
    dropout_layer4 = tf.nn.dropout(layer4, keep_prob)

    return tf.matmul(dropout_layer4, weights['out']) + biases['out']


def regularize(weights, biases):
    regularizers = tf.nn.l2_loss(weights['hidden1']) + \
                   tf.nn.l2_loss(biases['hidden1'])
    regularizers += tf.nn.l2_loss(weights['hidden2']) + \
                    tf.nn.l2_loss(biases['hidden2'])
    regularizers += tf.nn.l2_loss(weights['hidden3']) + \
                    tf.nn.l2_loss(biases['hidden3'])
    regularizers += tf.nn.l2_loss(weights['hidden4']) + \
                    tf.nn.l2_loss(biases['hidden4'])
    regularizers += tf.nn.l2_loss(weights['out']) + \
                    tf.nn.l2_loss(biases['out'])
    return regularizers


def main():
    # load the notMNIST data
    train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels = uda_tf_util.load_pickle('notMNIST.pickle')

    # reformat
    train_dataset, train_labels = uda_tf_util.reformat(train_dataset,
                                                       train_labels,
                                                       num_labels)
    valid_dataset, valid_labels = uda_tf_util.reformat(valid_dataset,
                                                       valid_labels,
                                                       num_labels)
    test_dataset, test_labels = uda_tf_util.reformat(test_dataset,
                                                     test_labels, num_labels)

    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # Tensor graph
    graph = tf.Graph()
    with graph.as_default():
        # Input dataset
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size,
                                                 image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        weights = {
            'hidden1': tf.Variable(
                tf.random_normal([n_input, n_hidden1],
                                 stddev=math.sqrt(2./image_size * image_size))),
            'hidden2': tf.Variable(
                tf.random_normal([n_hidden1, n_hidden2],
                                 stddev=math.sqrt(2./n_hidden1))),
            'hidden3': tf.Variable(
                tf.random_normal([n_hidden2, n_hidden3],
                                 stddev=math.sqrt(2./n_hidden2))),
            'hidden4': tf.Variable(
                tf.random_normal([n_hidden3, n_hidden4],
                                 stddev=math.sqrt(2./n_hidden3))),
            'out': tf.Variable(
                tf.random_normal([n_hidden4, n_classes],
                                 stddev=math.sqrt(2./n_hidden4)))
        }
        biases = {
            'hidden1': tf.Variable(tf.random_normal([n_hidden1])),
            'hidden2': tf.Variable(tf.random_normal([n_hidden2])),
            'hidden3': tf.Variable(tf.random_normal([n_hidden3])),
            'hidden4': tf.Variable(tf.random_normal([n_hidden4])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        keep_prob = tf.placeholder(tf.float32)
            
        logits = feedforward(tf_train_dataset, weights, biases, 0.8)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        )
        cost += 1e-4 * regularize(weights, biases)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(feedforward(tf_test_dataset,
                                                    weights, biases, 1.0))
        valid_prediction = tf.nn.softmax(feedforward(tf_valid_dataset,
                                                     weights, biases, 1.0))

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
                         tf_train_labels: batch_labels, keep_prob: 0.8}
            _, l, predictions = session.run(
                [optimizer,
                 cost,
                 train_prediction],
                feed_dict=feed_dict
            )

            if step % 1400 == 0:
                print 'Minibatch loss at step %d: %f' % (step, l)
                print 'Minibatch accuracy: %.1f%%' % accuracy(predictions,
                                                              batch_labels)
                print 'Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(),
                    valid_labels
                )
        print 'Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),
                                                 test_labels)

if __name__ == '__main__':
    main()
