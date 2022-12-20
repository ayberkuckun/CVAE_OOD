"""This module implements functions to calculate bias corrections per image adopting
the analytic (par.3.1) and algorithmic (par.3.2) approaches
"""
from collections import defaultdict

import numpy as np
import scipy.optimize
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm


def get_bias_corrected_lkl(decoder_dist, test_set, correction_func=None, pix_corrections=None):
    if decoder_dist == "cBern":
        test_set = tf.cast(tf.math.round(np.clip(test_set, 1e-3, 1-1e-3) * 1000), dtype=tf.int32)
        correction_set = tf.reduce_sum(correction_func(test_set), axis=[3, 2, 1])

        return correction_set

    elif decoder_dist == "cat":
        correction_func = np.vectorize(lambda x: pix_corrections[x])
        avg_px_val_set = tf.math.round(tf.reduce_mean(test_set, axis=[3, 2, 1]))

        correction_set = tf.squeeze(correction_func(avg_px_val_set))

        return correction_set

    else:
        raise ValueError("Decoder Distribution Not supported!")


def get_correction_func():
    px_values = np.linspace(1e-3, 1 - 1e-3, 999, dtype=np.float32)

    correction_dict = {}
    for px_value in tqdm(px_values):
        correction_dict[(px_value * 1000).round().astype(np.int32)] = analytical_bias_correction(px_value)

    return np.vectorize(lambda x: correction_dict[x])


def calculate_image_corrections(image, pix_corrections):
    corrections = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

    no = 0
    for k in range(image.shape[-1]):
        for i in range(256):
            mask = tf.boolean_mask(image[..., k], image[..., k] == i)

            if len(mask) > 0:
                corrections = corrections.write(no, float(len(mask)) * pix_corrections[i, k])
                no += 1

    correction = tf.reduce_mean(corrections.stack())

    return correction


def continuous_bernoulli_nll(lambdas, x):
    if 0.5 - 1e-6 < lambdas < 0.5 + 1e-6:
        c = 2
    else:
        c = 2 * np.arctanh(1 - 2 * lambdas) / (1 - 2 * lambdas)

    pdf = c * (lambdas**x) * ((1 - lambdas)**(1 - x))

    return -np.log(pdf)


def analytical_bias_correction(px_value):
    argmin = scipy.optimize.fmin(continuous_bernoulli_nll, 0.5, args=(px_value,), disp=False)[0]

    perfect_rc_ll = -continuous_bernoulli_nll(argmin, px_value)

    return np.float32(perfect_rc_ll)


def algorithmic_bias_correction(cvae, training_set):
    num_samples = cvae.num_samples
    cvae.num_samples = 1

    nc = training_set.shape[3]

    # corrections = np.zeros((256, nc), dtype=np.float32)

    A = defaultdict(lambda: defaultdict(list))

    for image in tqdm(training_set[:500]):
        image = tf.expand_dims(image, axis=0)

        reconstruction = cvae(image / 255.0, training=False)["reconstruction"]

        lp_x_z = tfp.distributions.Categorical(logits=reconstruction).log_prob(image)

        for k in range(nc):
            for v in range(256):
                b = tf.boolean_mask(lp_x_z[..., k], image[..., k] == v)
                if len(b) > 0:
                    A[k][v].append(tf.reduce_logsumexp(b) - tf.math.log(float(len(b))))

    C = defaultdict(lambda: 0)

    for k in range(nc):
        for v in range(256):
            C[v] += tf.reduce_logsumexp(A[k][v]) - tf.math.log(float(len(A[k][v])))
            # corrections[v, k] = tf.reduce_logsumexp(A[k][v]) - tf.math.log(float(len(A[k][v])))

    for v in range(256):
        C[v] /= nc

    cvae.num_samples = num_samples

    return C
    # return corrections
