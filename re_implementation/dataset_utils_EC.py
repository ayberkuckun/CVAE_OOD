"""Module containing functions for loading data sets and performing contrast
normalization as described in paragraph 3.3.

All dataset return normalised images with shape (32 x 32 x nc), nc = 1 for greyscale
and 3 for RGB datasets.

- Greyscale datasets: MNIST, FMNIST, EMNIST + Gray Noise
- Natural datasets: CIFAR10, SVHN_cropped, GTSRB + Color Noise
(CelebA gets a NonMatchingChecksumError which I have not been able to fix)

To get the same training/validation split as in the paper use frac = 0.9 in the functions below.
The noise generator simply generates 1000 samples of required noise (greyscale or color).
"""

import csv
import cv2
import os
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from matplotlib.image import imread
from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import opendatasets as od
import pandas as pd
import tensorflow_probability as tfp


# todo check if categorical labels explained in paper.
# todo check transform_to_dataset() func.
# todo use this example as reference to prepare datasets.

"""def getValData(dataset):
    if dataset == "mnist":
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist", split=['train[:90%]', 'train[90%:]', 'test'], with_info=True)

        ds_val = ds_val.map(
            partial(preprocess, inverted=False, mode='grayscale',
                    normalize=_NORMALIZE.value, dequantize=False,
                    visible_dist='cont_bernoulli'),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.cache()
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)"""


def transform_to_dataset(x_train, x_val, x_test):
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    return train_dataset, val_dataset, test_dataset


# @tf.function # if uncommented, CS is faster but noise doesn't work.
def get_dataset(dataset, decoder_dist, dataset_type, contrast_normalize=False, training=True):
    if dataset_type == "grayscale":
        if dataset == "mnist":
            (train_images, _), (val_images, _), (test_images, _) = load_mnist(decoder_dist, training=training)

        elif dataset == "fmnist":
            (train_images, _), (val_images, _), (test_images, _) = load_fmnist(decoder_dist, training=training)

        elif dataset == "emnist":
            (train_images, _), (val_images, _), (test_images, _) = load_emnist(decoder_dist, training=training)

        elif dataset == "noise":
            train_images = None
            val_images = None
            test_images = noise(purpose="test", type=dataset_type, dist=decoder_dist)

        else:
            raise NotImplementedError

    elif dataset_type == "natural":
        if dataset == "cifar10":
            (train_images, _), (val_images, _), (test_images, _) = load_cifar10(decoder_dist, training=training)

        elif dataset == "svhn":
            (train_images, _), (val_images, _), (test_images, _) = load_svhn(decoder_dist, training=training)

        elif dataset == "gtsrb":
            (train_images, _), (val_images, _), (test_images, _) = load_gtrsb(decoder_dist, training=training)

        elif dataset == "noise":
            train_images = None
            val_images = None
            test_images = noise(purpose="test", type=dataset_type, dist=decoder_dist)

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


    if contrast_normalize and dataset != "noise":
        train_images = tf.map_fn(contrast_normalization, train_images, back_prop=False, parallel_iterations=20)
        val_images = tf.map_fn(contrast_normalization, val_images, back_prop=False, parallel_iterations=20)
        test_images = tf.map_fn(contrast_normalization, test_images, back_prop=False, parallel_iterations=20)

        if decoder_dist == "cat":
            train_images = tf.round(train_images * 255.0)
            val_images = tf.round(val_images * 255.0)
            test_images = tf.round(test_images * 255.0)

    return train_images, val_images, test_images[:10000]


def resize(image_set):
    """Resizes image set to 32 x 32"""
    resized_output = np.expand_dims(image_set, axis=-1)
    resized_output = tf.image.resize(resized_output, [32, 32])
    return resized_output


def noise(purpose, type, dist):
    """Noise Generator
  Args:
    purpose: "train", "val" or "test"
    type: "grayscale" or "color"
    dist: "cBern" or "cat"
  Returns:
    output: list of 10000 noise images
  """
    if purpose == 'train':
        np.random.seed(100)

    if purpose == 'val':
        np.random.seed(200)
    else:
        np.random.seed(300)

    output = []

    if type == 'grayscale':
        output = [np.random.randint(low=0, high=256, size=(32, 32, 1)) for i in range(10000)]
    else:
        output = [np.random.randint(low=0, high=256, size=(32, 32, 3)) for i in range(10000)]

    output = tf.convert_to_tensor(output, dtype=tf.float32, dtype_hint=None, name=None)

    if dist =="cat":
        return output

    elif dist == "cBern":
        output = output / 256.0

    else:
        raise NotImplementedError

    return output


def load_cifar10(decoder_dist, frac=0.9, training=False):
    """Load CIFAR10 dataset and create training, validation and test sets.
  Args:
    decoder_dist: "cBern" or "cat"
    frac: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    print("The CIFAR dataset is being downloaded")
    (train_and_val_images, train_and_val_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    print("The pixel values are normalized to be between 0 and 1")

    if decoder_dist == "cBern":
        train_and_val_images, test_images = train_and_val_images / 255.0, test_images / 255.0

    elif decoder_dist == "cat":
        train_and_val_images, test_images = tf.cast(train_and_val_images, tf.float32), tf.cast(test_images, tf.float32)

    else:
        raise NotImplementedError

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    n = len(train_and_val_labels)
    cut = int(n * frac)
    train_images = train_and_val_images[0:cut]
    train_labels = train_and_val_labels[0:cut]

    val_images = train_and_val_images[cut:]
    val_labels = train_and_val_labels[cut:]

    #train_images, val_images, test_images = transform_to_dataset(train_images, val_images, test_images)

    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32, dtype_hint=None, name=None)
    # max = tf.math.reduce_max(train_images,axis=None, keepdims=False, name=None)
    # print("MAX",max)

    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32, dtype_hint=None, name=None)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32, dtype_hint=None, name=None)

    #return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    return (train_images, None), (val_images, None), (test_images, None)


def load_mnist(decoder_dist, frac=0.9, training=False):
    """Load mnist dataset and create training, validation and test sets.
  Args:
    decoder_dist: "cBern" or "cat"
    frac: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    print("The mnist dataset is being downloaded")
    (train_and_val_images, train_and_val_labels), (test_images, test_labels) = datasets.mnist.load_data()

    n = len(train_and_val_labels)
    cut = int(n * frac)
    train_images = train_and_val_images[0:cut]
    #train_labels = train_and_val_labels[0:cut]

    val_images = train_and_val_images[cut:]
    #val_labels = train_and_val_labels[cut:]

    if decoder_dist == "cBern":
        # Resize & Normalize images
        train_images = resize(train_images) / 255.0
        val_images = resize(val_images) / 255.0
        test_images = resize(test_images) / 255.0

    elif decoder_dist == "cat":
        # Resize & Normalize images
        train_images = resize(train_images)
        val_images = resize(val_images)
        test_images = resize(test_images)

    else:
        raise NotImplementedError

    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32, dtype_hint=None, name=None)
    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32, dtype_hint=None, name=None)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32, dtype_hint=None, name=None)

    return (train_images, None), (val_images, None), (test_images, None)


def load_fmnist(decoder_dist, frac=0.9, training=False):
    """Load fashion_mnist dataset and create training, validation and test sets.
  Args:
    decoder_dist: "cBern" or "cat"
    frac: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    print("The fmnist dataset is being downloaded")
    (train_and_val_images, train_and_val_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    n = len(train_and_val_labels)
    cut = int(n * frac)
    train_images = train_and_val_images[0:cut]
    #train_labels = train_and_val_labels[0:cut]

    val_images = train_and_val_images[cut:]
    #val_labels = train_and_val_labels[cut:]

    if decoder_dist == "cBern":
        # Resize & Normalize images
        train_images = resize(train_images) / 255.0
        val_images = resize(val_images) / 255.0
        test_images = resize(test_images) / 255.0

    elif decoder_dist == "cat":
        # Resize & Normalize images
        train_images = resize(train_images)
        val_images = resize(val_images)
        test_images = resize(test_images)

    else:
        raise NotImplementedError

    #train_images, val_images, test_images = transform_to_dataset(train_images, val_images, test_images)
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32, dtype_hint=None, name=None)
    # max = tf.math.reduce_max(train_images,axis=None, keepdims=False, name=None)
    # print("MAX",max)

    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32, dtype_hint=None, name=None)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32, dtype_hint=None, name=None)

    #return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    return (train_images, None), (val_images, None), (test_images, None)


def load_emnist(decoder_dist, frac=0.9, training=False):
    """Load e_mnist letters dataset and create training, validation and test sets.
  Args:
    decoder_dist: "cBern" or "cat"
    split: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    print("The emnist dataset is being downloaded")
    train_and_val_images, train_and_val_labels = extract_training_samples('letters')
    test_images, test_labels = extract_test_samples('letters')

    n = len(train_and_val_labels)
    cut = int(n * frac)
    train_images = train_and_val_images[0:cut]
    #train_labels = train_and_val_labels[0:cut]

    val_images = train_and_val_images[cut:]
    #val_labels = train_and_val_labels[cut:]

    # max = tf.math.reduce_max(test_images,axis=None, keepdims=False, name=None)
    # print("MAX",max)

    if decoder_dist == "cBern":
        # Resize & Normalize images
        train_images = resize(train_images) / 255.0
        val_images = resize(val_images) / 255.0
        test_images = resize(test_images) / 255.0

    elif decoder_dist == "cat":
        # Resize & Normalize images
        train_images = resize(train_images)
        val_images = resize(val_images)
        test_images = resize(test_images)

    else:
        raise NotImplementedError

    #train_images, val_images, test_images = transform_to_dataset(train_images, val_images, test_images)
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32, dtype_hint=None, name=None)
    # max = tf.math.reduce_max(train_images,axis=None, keepdims=False, name=None)
    # print("MAX",max)

    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32, dtype_hint=None, name=None)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32, dtype_hint=None, name=None)

    #return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    return (train_images, None), (val_images, None), (test_images, None)

def load_svhn(decoder_dist, frac=0.9, training=False):
    """Load svhn_cropped dataset and create training, validation and test sets.
  Args:
    decoder_dist: "cBern" or "cat"
    split: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    print("The SVHN_cropped dataset is being downloaded")
    train_images = tfds.load('svhn_cropped', split='train[:90%]', shuffle_files=True)
    val_images = tfds.load('svhn_cropped', split='train[90%:]', shuffle_files=True)
    test_images = tfds.load('svhn_cropped', split='test', shuffle_files=True)

    # train_and_val_set = tfds.load('svhn_cropped', split='train', shuffle_files=True)
    # test = tfds.load('svhn_cropped', split='test', shuffle_files=True)

    # train_and_val_images = [item["image"].numpy() for item in train_and_val_set.take(-1)]
    #train_and_val_labels = [item["label"].numpy() for item in train_and_val_set.take(-1)]

    # test_images = [item["image"].numpy() for item in test.take(-1)]
    #test_labels = [item["label"].numpy() for item in test.take(-1)]

    # n = len(train_and_val_images)
    # cut = int(n * frac)

    # train_images = train_and_val_images[0:cut]
    #train_labels = train_and_val_labels[0:cut]

    # val_images = train_and_val_images[cut:]
    #val_labels = train_and_val_labels[cut:]

    test_images = test_images.map(lambda x: x["image"])
    train_images = train_images.map(lambda x: x["image"])
    val_images = val_images.map(lambda x: x["image"])

    test_images = tf.convert_to_tensor(list(test_images), dtype=tf.float32, dtype_hint=None, name=None)
    train_images = tf.convert_to_tensor(list(train_images), dtype=tf.float32, dtype_hint=None, name=None)
    val_images = tf.convert_to_tensor(list(val_images), dtype=tf.float32, dtype_hint=None, name=None)

    #max = tf.math.reduce_max(test_images,axis=None, keepdims=False, name=None)
    #print("MAX", max)

    if decoder_dist == "cBern":
        # Normalize images (already 32 x32)
        test_images = test_images / 255.0
        train_images = train_images / 255.0
        val_images = val_images / 255.0

    elif decoder_dist == "cat":
        # Normalize images (already 32 x32)
        test_images = test_images
        train_images = train_images
        val_images = val_images

    else:
        raise NotImplementedError

    #train_images, val_images, test_images = transform_to_dataset(train_images, val_images, test_images)
    #train_images = tf.convert_to_tensor(train_images, dtype=tf.float32, dtype_hint=None, name=None)
    # max = tf.math.reduce_max(train_images,axis=None, keepdims=False, name=None)
    # print("MAX",max)

    #val_images = tf.convert_to_tensor(val_images, dtype=tf.float32, dtype_hint=None, name=None)
    #test_images = tf.convert_to_tensor(test_images, dtype=tf.float32, dtype_hint=None, name=None)

    #return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    return (train_images, None), (val_images, None), (test_images, None)


def load_celebA(frac=0.9):
    """Load celeb_a dataset and create training, validation and test sets.

  Args:
    decoder_dist: "cBern" or "cat"
    frac: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    print("The CelebA dataset is being downloaded")
    train_and_val_set = tfds.load('celeb_a', split='train', shuffle_files=True)
    test = tfds.load('celeb_a', split='test', shuffle_files=True)

    train_and_val_images = [item["image"].numpy() for item in train_and_val_set.take(-1)]
    train_and_val_labels = [item["label"].numpy() for item in train_and_val_set.take(-1)]

    test_images = [item["image"].numpy() for item in test.take(-1)]
    test_labels = [item["label"].numpy() for item in test.take(-1)]

    n = len(train_and_val_labels)
    cut = int(n * frac)

    train_images = train_and_val_images[0:cut]
    train_labels = train_and_val_labels[0:cut]

    val_images = train_and_val_images[cut:]
    val_labels = train_and_val_labels[cut:]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def load_gtrsb(decoder_dist, frac=0.9, training=False):
    """Load gtrsb dataset from the given path and create training, validation and test sets.
  Source: https://medium.com/analytics-vidhya/cnn-german-traffic-signal-recognition-benchmarking-using-tensorflow-accuracy-80-d069b7996082

  Args:
    decoder_dist: "cBern" or "cat"
    frac: fraction of training data used for training
  Return:
    ds_train: training set
    ds_val: validation set
    ds_test: test set
  """
    dataset_url = 'https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'
    od.download(dataset_url)

    path = './gtsrb-german-traffic-sign/'
    meta_df = pd.read_csv('./gtsrb-german-traffic-sign/Meta.csv')
    test_data = pd.read_csv('./gtsrb-german-traffic-sign/Test.csv')
    train_data = pd.read_csv('./gtsrb-german-traffic-sign/Train.csv')

    train_and_val_part_1 = train_data["Path"].values
    train_and_val_image_paths = [path + part for part in train_and_val_part_1]
    train_and_val_labels = train_data["ClassId"].values

    test_part_1 = test_data["Path"].values
    test_image_paths = [path + part for part in test_part_1]
    #test_labels = test_data["ClassId"].values

    train_and_val_images = [cv2.imread(elem) for elem in train_and_val_image_paths]
    train_and_val_images = [tf.image.resize(image, [32, 32]) for image in train_and_val_images]

    n = len(train_and_val_labels)
    cut = int(n * frac)

    train_images = train_and_val_images[0:cut]
    #train_labels = train_and_val_labels[0:cut]

    val_images = train_and_val_images[cut:]
    #val_labels = train_and_val_labels[cut:]

    test_images = [cv2.imread(elem) for elem in test_image_paths]
    test_images = [tf.image.resize(image, [32, 32]) for image in test_images]

    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32, dtype_hint=None, name=None)
    # max = tf.math.reduce_max(train_images,axis=None, keepdims=False, name=None)
    # print("MAX",max)

    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32, dtype_hint=None, name=None)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32, dtype_hint=None, name=None)

    if decoder_dist == "cBern":
        train_images = train_images / 255
        val_images = val_images / 255
        test_images = test_images / 255

    elif decoder_dist == "cat":
        train_images = train_images
        val_images = val_images
        test_images = test_images

    else:
        raise NotImplementedError

    #return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    return (train_images, None), (val_images, None), (test_images, None)


@tf.function(input_signature=[tf.TensorSpec(shape=[32, 32, None], dtype=tf.float32)])
def contrast_normalization(image):
    """Function which performs image normalization as described in par.3.3
    Args:
      image: image to be normalised
    Return:
      image: normalised image"""

    a = tfp.stats.percentile(image, 5)
    r = tfp.stats.percentile(image, 95) - a

    output = (image - a) / r
    output = tf.clip_by_value(output, 0., 1.)

    # output = tf.map_fn(lambda x: tf.math.minimum(tf.math.maximum(0.0, (x - a) / r), 1.0), image, back_prop=False, parallel_iterations=20)

    # output = [tf.math.minimum(tf.math.maximum(0, (x - a) / r), 1) for x in image]
    # normalised = tf.convert_to_tensor(output, dtype=tf.float32, dtype_hint=None, name=None)

    return output
