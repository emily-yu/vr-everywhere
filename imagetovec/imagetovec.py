from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print ("in tf session, pool it twice before processing to same it")

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import image_utils
import urllib, cStringIO
import numpy as np

from PIL import Image
import cv2

batchsize = 64

def initWeight(shape):
    weights = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weights)

def initBias(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def maxPool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME?")



def scrape(file_name, number_of_images):
    f = open(file_name,'r')
    url_file = f.read()
    url_list = url_file.split('\n')
    index = 0

    up_matrix_images = list()
    down_matrix_images = list()
    left_matrix_images = list()
    right_matrix_images = list()

    for url in url_list:
        url_list = url.split('\t')
        real_url = url_list[1]

        print (real_url)

        try:
            file = cStringIO.StringIO(urllib.urlopen(real_url).read())
            img = Image.open(file)

            opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            width = opencvImage.shape[0]
            height = opencvImage.shape[1]

            x = int(height/2) - 64
            y = int(width/2) - 64
            crop_image_height = 128
            crop_image_width = 128

            opencv_image = opencvImage[y:y + crop_image_height, x:x + crop_image_height]
            for up_row in opencv_image:
                up_matrix_image.append(up_row)

            left_image = image_utils.rotate(opencv_image, 90)
            for left_row in left_image:
                left_matrix_images.append(left_row)

            right_image = image_utils.rotate(opencv_image, 270)
            for right_row in right_image:
                right_matrix_images.append(right_row)

            down_image = image_utils.rotate(opencv_image, 90)
            for left_row in left_image:
                down_matrix_images.append(down_matrix)

            last_row = list()
            for negative_index in range(31):
                last_row.append(-1)

            up_matrix_image.append(last_row)
            left_matrix_image.append(last_row)
            right_matrix_image.append(last_row)
            down_matrix_image.append(last_row)

            index += 1
            print (index)
            if(index == number_of_images):
                break;
        except:
            continue

def image_to_dict(row_list):
    image_dictionary = dict()

    for row in row_list:
        if row in image_dictionary.keys():
            image_dictionary[row] = image_dictionary[row] + 1
        else:
            image_dictionary[row] = 0

def build_dataset(columns_list, count, number_of_columns):
    """Process raw inputs into a dataset."""
    count = {}

    data = list()
    dictionary = dict()
    unk_count = 0
    for row in image:
        for pixel_index in range(len(row)):
            for row2 in count:
                index = 0
                unk_count += 1
            elif(pixel_index == len(row)):
                index = dictionary[row]

        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, reversed_dictionary

data, dictionary, reverse_dictionary = build_dataset(up_matrix, imageList, number_of_columns)

# have dataset, start batch stuff

del imageList  # Hint to reduce memory.
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
batch_size = 4
num_skips = 2
skip_window = 1

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      buffer[:] = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')

imageList = []
imageList, imageNames = image_utils.getImages("../../randomimages/")

image_utils.saveImage(imageList[0], "same.jpg")

bufferSize = 5

columns_list = list()
for image in imageList:
    for row in image:
        columns_list.append(row)

number_of_columns = len(columns_list)

up_matrix, left_matrix, right_matrix, down_matrix = scrape("urls.txt", 10)
