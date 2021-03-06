#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

filename = 'text8.zip'
data_index = 0

# Parameters
vocabulary_size = 50000
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.
num_steps = 100001

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name)).split()
    f.close()


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, count, dictionary, reverse_dictionary

def main():
    words = read_data(filename)
    print('Data size %d' % len(words))

    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.

    def generate_batch(batch_size, num_skips, skip_window):
        global data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        return batch, labels

    batch, labels = generate_batch(batch_size=8,
                                   num_skips=2,
                                   skip_window=1)
    for i in range(8):
        print('%d -> %d' % (batch[i], labels[i, 0]))
        print('%s -> %s' % (reverse_dictionary[batch[i]],
                            reverse_dictionary[labels[i, 0]]))

    graph = tf.Graph()
    with graph.as_default():
        # Input data
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Variables
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Model
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(softmax_weights,
                                       softmax_biases,
                                       embed,
                                       train_labels,
                                       num_sampled,
                                       vocabulary_size)
        )
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # Compute the similarity between minibatch examples and all embeddings
        # We use the cosine distance.
        norm = tf.sqrt(
            tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)
        )
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(valid_embeddings,
                               tf.transpose(normalized_embeddings))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'Initialized.'
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = generate_batch(batch_size,
                                                      num_skips,
                                                      skip_window)
            feed_dict = {train_dataset: batch_data,
                         train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    print 'Average loss at step %d: %f' % (step,
                                                           average_loss)
                    average_loss = 0
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print log
        final_embeddings = normalized_embeddings.eval()

if __name__ == '__main__':
    main()
