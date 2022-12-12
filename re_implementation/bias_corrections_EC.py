"""This module implements functions to calculate bias corrections per image adopting
the analytic (par.3.1) and algorithmic (par.3.2) approaches
"""
import numpy as np
from scipy.optimize import minimize
from numpy.random import rand
import tensorflow as tf
import itertools
import tensorflow_probability as tfp

def get_bias_corrected_lkl(cvae, image, training_set):
    """Returns the bias Correction per image for Continuous Bernoulli and categorical
    visible distributions.
    Args:
    -cvae: trained model
    -image: test image which requires bias correction, array((R,C,nc), dtype=float)),
    R = number of raws, C = number of columns, nc = number of channels
    -training_set: set of images used for training cvae, array((D,R,C,nc), dtype=float)),
    with D = number of images
    Returns:
    -correction: (scalar)likelyhood correction (per image) calculated with analytical
    (for "cBern") or algorithmical (for "cat") approach
    """
    #print("VERIFICATION", image.shape)

    r = image.shape[1]
    c = image.shape[2]
    nc = image.shape[3]

    if cvae.decoder_dist == "cBern":
        lamdas = tf.math.sigmoid(cvae.predict(image)["reconstruction"])

        #if tf.math.reduce_min(lamdas)<=0:
        #    print("ERROR: negative or zero lamda!")
        #if tf.math.reduce_max(lamdas)>=1:
        #    print("ERROR: too big lamda!")

        lamdas = tf.clip_by_value(lamdas, 1e-10, 1-1e-10, name=None)
        MIN = tf.math.reduce_min(lamdas)
        MAX = tf.math.reduce_max(lamdas)

        #print('LAMDA', lamdas.shape)
        #print("MIN", MIN)
        #print("MAX", MAX)

        correction = analytical_bias_correction(lamdas)

    elif cvae.decoder_dist == "cat":
        pix_corrections = algorithmic_bias_correction(cvae, training_set)
        corrections = np.zeros((r, c, nc), dtype=float)

        for k in range(nc):
            for i in range(r):
                for j in range(c):
                    x = int(image[0][i][j][k])
                    corrections[i,j,k] = pix_corrections[x][k]

        #for k in range(nc):
        #    for i in range(256):
        #        corrections[..., k][(output[..., k] == i).numpy()] = pix_corrections[i, k]
        correction = np.mean(corrections)

    else:
        print("Decoder Distribution Not supported!")
    return correction

def decoded_pix(lmd):
    """Calculation of decoded pixel when the value of lmd is different from 0.5
    (formula at the bottom of the left column in page 3)"""

    #if tf.math.reduce_min(1 - 2 * lmd)<= -1:
    #    print("ERROR: arctanh not defined for values < -1")
    #if tf.math.reduce_max(1 - 2 * lmd)>=1:
    #    print("ERROR: arctanh not defined for values > 1")

    elem_1 = np.divide(lmd, 2 * lmd - 1)
    elem_2 = np.divide(1, 2 * np.arctanh(1 - 2 * lmd))
    return elem_1 + elem_2

def C(lmd):
    """Calculation of C(lmd), see fomula at
    https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution"""
    return np.divide(2 * np.arctanh(1 - 2 * lmd), 1 - 2 * lmd)

def NRE_perfect(lamdas):
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
    """Bias Correction (per image) for Continuous Bernoulli visible distributions
  described in paragraph 3.1.
  Args:
    -lambdas: shape parameters of the visbile Beurnoulli decoder for the image,
    expected np.array((r,c,nc),dtype= float) r = number of pixel raws of
    image, c = number of pixel columns of image, nc = number of channels.
  Returns:
    -evaluation: (scalar) correction for the likelyhood (per image) calculated with formula (3)
  """
    # Minimization of NRE with the respect of lmd using Nelder-Mead
    # Source: https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/
    r = lamdas.shape[0]
    c = lamdas.shape[1]
    nc = lamdas.shape[2]
    #lmd_min, lmd_max = np.zeros((r, c, nc)), np.ones((r, c, nc))
    lmd_min, lmd_max = 1e-10*np.ones((r, c, nc)), (1-1e-10)*np.ones((r, c, nc))
    # define the starting point as a random sample from the domain
    pt = lmd_min + np.random.rand(r, c, nc) * (lmd_max - lmd_min)
    # perform the search
    result = minimize(NRE_perfect, pt, method='nelder-mead', options={'maxiter': 1000, 'maxfev': 1000})
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    print('Solution: %s' % result['x'])
    # evaluate solution
    solution = result['x']
    evaluation = NRE_perfect(solution)
    #print('Solution: f(%s) = %.5f' % (solution, evaluation))
    return evaluation

def algorithmic_bias_correction(cvae, training_set):
    """Algorithmic Correction for Categorical visible distributions described in
     paragraph 3.2
  Args:
    -cvae: trained model
    -training set: set of images used for training cvae, array((D,R,C,nc), dtype=float)),
    D = number of images, R = number of raws, C = number of columns, nc = number of channels
  Returns:
    -log correction factor: correction matrix calculated with algorithm 1,
    array((256, nc), dtype= float)
  """
    if cvae.decoder_dist != "cat":
        print("Decoder Distribution Error!")

    D = training_set.shape[0]
    nc = training_set.shape[3]

    # Correction matrix for the data set
    #Correction = tf.zeros((256, nc), dtype=tf.float32)
    Correction = np.zeros((256, nc), dtype=float)

    # Correction matrix for the images
    #A = tf.zeros((256, nc), dtype=tf.float32)
    A = np.zeros((256, nc), dtype=float)

    for image in training_set:
        #Correction matrix for the pixels in an image
        B = np.zeros((256, nc), dtype=float)
        counter_B = np.ones((256, nc), dtype= float)
        image = tf.expand_dims(image, axis=0)
        #z = "forward pass of encoder on image"  # REPLACE string WITH CORRECT CLASS METHOD
        # "forward pass of decoder on the above z"  # REPLACE string WITH CORRECT CLASS METHOD
        decoded_img = cvae.predict(image)["reconstruction"]

        reconstruction = tf.reshape(decoded_img, (cvae.num_samples, -1, 32, 32, cvae.num_channel, 256))
        reconstruction = reconstruction.numpy()

        lp_x_z = tfp.distributions.Categorical(logits=reconstruction).log_prob(image)
        lp_x_z = lp_x_z.numpy()

        print("TEST LP_X_Z", lp_x_z.shape)
        print("TEST rec", reconstruction.shape)
        #print("TEST bis", lp_x_z)

        #Algorithm 1
        for k in range(nc):
            for v in range(256):
                #indexes = tf.where(reconstruction_2[:,:,k] == v)
                #B[v][k] = tf.sum(lp_x_z[indexes])
                #counter_B[v][k] += tf.set.size(indexes)
                indexes = np.where(np.argmax(reconstruction[:,:,:,:,k,:], axis = 4) == v)
                B[v][k] = np.sum(lp_x_z[indexes])
                counter_B[v][k] += len(indexes)

        #print("PROVA B Counter", B.shape, counter_B)
        #counter_B = tf.convert_to_tensor([x-1 for x in counter_B if x>1])
        #B = tf.convert_to_tensor(B)
            counter_B[:,k] = [x-1 for x in counter_B[:,k] if x>1]

        #A += tf.math.divide(B, counter_B)
        A += np.divide(B, counter_B)

    #maybe this is enough? amswer: see above
    #Correction = tf.reduce_mean(cvae.predict(training_set)["reconstruction"])

    #Correction = tf.math.divide(A, D)
    Correction = np.divide(A, D)
    Correction = tf.convert_to_tensor(Correction)
    return Correction
