"""This module implements functions to implement bias correction as described
in paragraph 3.1 and algorithmic correction as described in paragraph 3.2

OBS: PERHAPS THE ALGORITHMIC CORRECTION FUNCTION SHOULD BECOME A METHOD OF THE VAE CLASS,
AND STRINGS AT CODE LINES 95,96 102 SHOULD BE REPLACED WITH THE PROPER DEFINED CLASS METHODS
"""
import numpy as np
from scipy.optimize import minimize
from numpy.random import rand
import tensorflow as tf
import itertools


def decoded_pix(lmd):
    """Calculation of decoded pixel when the value of lmd is different from 0.5
    (formula at the bottom of the left column in page 3)"""
    elem_1 = np.divide(lmd, 2 * lmd - 1)
    elem_2 = np.divide(1, 2 * np.arctanh(1 - 2 * lmd))
    return elem_1 + elem_2


def C(lmd):
    """Calculation of C(lmd), see fomula at
    https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution"""
    return np.divide(2 * np.arctanh(1 - 2 * lmd), 1 - 2 * lmd)


def NRE(lamdas):
    """Negative reconstruction error (NRE) with perfect reconstruction
    calculated with formula (1)"""
    decoded_img = [0.5 if lmd == 0.5 else decoded_pix(lmd) for lmd in lamdas]
    decoded_img = tf.convert_to_tensor(decoded_img, dtype=tf.float32, dtype_hint=None, name=None)

    C_of_lamda = [2 if lmd == 0.5 else C(lmd) for lmd in lamdas]
    elem_1 = np.log(C_of_lamda)
    elem_2 = decoded_img * np.log(lamdas)
    elem_3 = (1 - decoded_img) * np.log(1 - lamdas)
    return np.sum(elem_1 + elem_2 + elem_3)


def analytical_bias_correction(lamdas):
    """Bias Correction for Continuous Bernoulli visible distributions
  described in paragraph 3.1.
  Args:
    -VAE_ML: uncorrected VAE likelyhood for a given image set, expected
    array((D,),dtype= float) with D = number of images in data set;
    -lambdas: shape parameters of the visbile Beurnoulli decoder for the data set,
    expected np.array((D,R,C),dtype= float) with D as above, R= number of pixel raws of
    image and C = number of pixel columns of image.
  Returns:
    -corrected_likelihood: corrected likelyhood calculated with formula (3)
  """
    # Minimization of NRE with the respect of lmd using Nelder-Mead
    # Source: https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/
    d = lamdas.shape[0]
    r = lamdas.shape[1]
    c = lamdas.shape[2]
    lmd_min, lmd_max = np.zeros((d, r, c)), np.ones((d, r, c))
    # define the starting point as a random sample from the domain
    pt = lmd_min + np.random.rand(d, r, c) * (lmd_max - lmd_min)
    # perform the search
    result = minimize(NRE, pt, method='nelder-mead', options={'maxiter': 1000, 'maxfev': 1000})
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = NRE(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))

    return evaluation


def algorithmic_bias_correction(cvae, training_set):
    """PERHAPS THIS FUNCTION SHOULD BECOME A METHOD OF THE VAE CLASS...
  STRINGS AT LINES 95,96 102 SHOULD BE REPLACED WITH THE DEFINED CLASS METHODS

  Algorithmic Correction for Categorical visible distributions
  described in paragraph 3.2
  Args:
    -training set: set of images used for training, array((D, R,C,nc), dtype=float)),
    D = number of images, R = number of raws, C = number of columns, nc = number of channels.
  Returns:
    -log correction factor: correction matrix calculated with algorithm 1,
    array((256, nc), dtype= float)
  """
    D = training_set.shape[0]
    nc = training_set.shape[3]

    # Correction matrix for the data set
    Correction = np.array((256, nc), dtype=float)

    # Correction matrix for the images
    A = np.array((256, nc), dtype=float)
    counter_A = np.zero(256, nc)

    for image in training_set:
        # Correction matrix for the pixels in an image
        B = np.array((256, nc), dtype=float)
        counter_B = np.zero(256, nc)

        z = "forward pass of encoder on image"  # REPLACE string WITH CORRECT CLASS METHOD
        decoded_x = cvae.predict(image)["reconstruction"] # "forward pass of decoder on the above z"  # REPLACE string WITH CORRECT CLASS METHOD

        for k in range(nc):
            for i in range(32):
                for j in range(32):
                    v = decoded_x[i][j][k]
                    B[v][
                        k] += "value of the decoder categorical distribution at the pixel [i][j][k]"  # REPLACE string WITH CORRECT CLASS METHOD
                    counter_B[v][k] += 1

        A += np.divide(B, counter_B)

        # maybe this is enough?
        A = tf.reduce_mean(cvae.predict(training_set)["reconstruction"])

    Correction = np.log(A / D)
    return Correction
