from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

    matrices = dict()

    matrices["up"] = list()
    matrices["left"] = list()
    matrices["right"] = list()
    matrices["down"] = list()

    for url in url_list:
        url_list = url.split('\t')
        real_url = url_list[1]

        try:
            file = cStringIO.StringIO(urllib.urlopen(real_url).read())
            img = Image.open(file)

            opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            width = opencvImage.shape[0]
            height = opencvImage.shape[1]

            x = int(height/2) - 64
            y = int(width/2) - 64
            crop_image_height = 32
            crop_image_width = 32

            opencv_image = opencvImage[y:y + crop_image_height, x:x + crop_image_height]

            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            image_utils.saveImage("up.jpg", opencv_image)

            for up_row in opencv_image:
                matrices["up"].append(up_row)

            left_image = image_utils.rotate(opencv_image, 90)

            image_utils.saveImage("left.jpg", left_image)

            for left_row in left_image:
                matrices["left"].append(left_row)

            right_image = image_utils.rotate(opencv_image, 270)
            image_utils.saveImage("right.jpg", right_image)
            for right_row in right_image:
                matrices["right"].append(right_row)

            down_image = image_utils.rotate(opencv_image, 180)
            image_utils.saveImage("down.jpg", down_image)
            for down_row in down_image:
                matrices["down"].append(down_row)


            # last_row = list()

            # for _ in range(0,31):
                # last_row.append(-1)

            # matrices["up"].append(last_row)
            # matrices["left"].append(last_row)
            # matrices["right"].append(last_row)
            # matrices["down"].append(last_row)
            print (index)
            index += 1
            if(index >= number_of_images):
                break;
        except:
            continue

    return matrices

def image_to_dict(row_list):
    image_dictionary = dict()
    row_to_index = dict()

    for row in row_list:
        inside = False
        for r in row_to_index.values():
            if all(r[x] == row[x] for x in range(len(row) - 1)):
                inside = True
                key = 0
                for tempkey, value in row_to_index.iteritems():
                    if all(value[x] == row[x] for x in range(len(row) - 1)):
                        key = tempkey
                image_dictionary[key] = image_dictionary[key] + 1
                break;

        if not inside:
            image_dictionary[len(row_to_index)] = 1
            row_to_index[len(row_to_index)] = row

    return image_dictionary, row_to_index

def build_dataset(count, row_to_index):
    """Process raw inputs into a dataset."""
    # count row to index do shit, check if in that instead

    data = list()
    dictionary = dict()
    dictionary_sketch = {}
    unk_count = 0

    for x in range(len(row_to_index.values())):
        current_row2index = row_to_index.values()[x]
        row_to_index.values()[x] = list(current_row2index)

    # for key in count:
    for key,value in row_to_index.iteritems():
        # print ("KEY is {0}".format(key))
        # for current_word_key, realkey in count.iteritems():
            # print ("currnet word is {0}".format(current_word_key))
            # if(key == current_word_key):
            # print (realkey)
        dictionary[key] = len(dictionary.keys())
        dictionary_sketch[key] = list(value)

    print ("dictionary_sketch is {0}".format(dictionary_sketch))

    data = list()
    unk_count = 0

    for row in row_to_index.values():
        same = False
        for v in dictionary_sketch.values():
            if all(row[x] == v[x] for x in range(len(row) - 1)):
                same = True
            index = dictionary[key]
        if(same == False):
            index = 0
        data.append(index)

    reversed_dictionary = dict(zip(dictionary.values(), dictionary_sketch.values()))
    # print ("the stuff ishabhha")
    # print (dictionary.values())
    # print (dictionary_sketch.values())
    # print (reversed_dictionary)
    return data, dictionary, reversed_dictionary

images_num = 890

matrices = scrape("urls.txt", images_num)

data_dict = dict()
dictionary_dict = dict()
reverse_dict = dict()

for direction in ['up', 'left', 'right', 'down']:
    count, row_to_index = image_to_dict(matrices[direction])
    data, dictionary, reverse = build_dataset(count, row_to_index)

    data_dict[direction] = data
    dictionary_dict[direction] = dictionary
    reverse_dict[direction] = reverse

del matrices  # Hint to reduce memory.

data_index = 0
batch_size = 128
num_skips = 4
skip_window = 1
up_size = len(reverse_dict["up"])
left_size = len(reverse_dict["left"])
right_size = len(reverse_dict["right"])
down_size = len(reverse_dict["down"])

vocabulary_size = 283
print ("up siez is {0}".format(up_size))
print ("left isze is {0}".format(left_size))
print ("right siez is {0}".format(right_size))
print ("down siez is {0}".format(down_size))
print (vocabulary_size)


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window, direction):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data_dict[direction]):
        data_index = 0
    # print (data_dict[direction])
    buffer.extend(data_dict[direction][data_index:data_index + span])
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

        if data_index == len(data_dict[direction]):
            buffer = data_dict[direction][:span]
            data_index = span
        else:
            buffer.append(data_dict[direction][data_index])
            data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data_dict[direction]) - span) % len(data_dict[direction])
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, direction="up")


# for i in range(8):
  # print(batch[i], reverse_dictvocabulary_size['up'][batch[i]],
        # '->', labels[i, 0], reverse_dict['up'][labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 32  # Dimension of the embedding vector.
skip_window = 4       # How many words to consider left and right.
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
    print (vocabulary_size)
    print (embedding_size)
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
  print ("yahallo marker thing")
  print (norm)
  print (embeddings)
  normalized_embeddings = embeddings / norm
  print (normalized_embeddings)
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

for direction in directions["up", "left", "down", "right"]:
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      saver = tf.train.Saver()

      # for direction in ["up", "right", "left", "down"]:
      init.run()

      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window, direction)

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
            valid_word = reverse_dict[direction][valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            print ("similarity is {0}".format(sim))
            print ("nearest is {0}".format(nearest))

            log_str = 'Nearest to %s:' % valid_word
            for k in xrange(top_k):
              print (len(reverse_dict[direction]))
              close_word = reverse_dict[direction][nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
          saver.save(session, "/Users/kevin/documents/imagetovec/" + direction + ".ckpt", global_step=i)
      final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.
