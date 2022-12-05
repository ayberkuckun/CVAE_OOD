import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(dataset, decoder_dist):
    if dataset == "mnist":
        ds_train = tfds.load('mnist', split='train')

        if decoder_dist == "cBern":
            ds_train = ds_train.map(process)

        elif decoder_dist == 'cat':
            ds_train = ds_train.map(process_cat)

        else:
            raise ValueError("Undefined Decoder Output Distribution.")

        return np.array(list(ds_train))

    else:
        return None, None


def process(image):
    image = tf.cast(tf.image.resize(image["image"], [32, 32], antialias=True), tf.float32) / 255
    return image


def process_cat(image):
    image = tf.cast(tf.image.resize(image["image"], [32, 32], antialias=True), tf.float32)
    return image


