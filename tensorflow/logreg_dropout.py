#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import uda_tf_util

# Parameters
num_labels = 10
image_size = 28
batch_size = 128
lambda_2 = 0.001
training_epochs = 15
learning_rate = 0.001
num_steps = 8001

# Network parameters
n_hidden = 1024
n_input = image_size * image_size
n_classes = num_labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def multilayer_perceptron(X, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['hidden']), biases['hidden']))
    return tf.add(tf.matmul(layer1, weights['out']), biases['out'])


def main():
    # load the notMNIST data
    train_dataset, train_labels, valid_dataset, valid_labels, \
        test_dataset, test_labels = uda_tf_util.load_pickle('notMNIST.pickle')

    # reformat
    train_dataset, train_labels = uda_tf_util.reformat(train_dataset,
                                                       train_labels, num_labels)
    valid_dataset, valid_labels = uda_tf_util.reformat(valid_dataset,
                                                       valid_labels, num_labels)
    test_dataset, test_labels = uda_tf_util.reformat(test_dataset,
                                                     test_labels, num_labels)

    print('Training', train_dataset.shape, train_labels.shape)
    print('Test', test_dataset.shape, test_labels.shape)
    print('Validation', valid_dataset.shape, valid_labels.shape)

    # Tensor graph
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size,
                                                 image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([image_size * image_size,
                                                   num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        )
        loss += lambda_2 * tf.nn.l2_loss(weights)

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights) + biases)

    # Graph computation
    num_steps = 8001
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'Initialized.'

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset+batch_size), :]
            batch_labels = train_labels[offset:(offset+batch_size), :]
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer,
                 loss,
                 train_prediction],
                feed_dict=feed_dict
            )

            if step % 500 == 0:
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
