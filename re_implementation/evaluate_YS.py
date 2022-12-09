
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


"""The evaluate(
    args:
    -x=None,input data
    -y=None,target data
    -batch_size=None, 
    -sample_weight=None, for weighted loos function
    -steps=None, sample batch
    -callbacks=None, 
    -return_dict=False, return the list
"""

def evaluating(cvae, x_val):

    # Evaluate on Validation data
    cvae.evaluate(x_val)

    # Computes an importance weighted log likelihood estimate.
    in_sample = next(iter(x_val))
    in_sample = np.expand_dims(in_sample, axis=0)
    cvae.num_samples = 100
    output = cvae.predict(in_sample)
    cvae.appy_correction = True
    loss1 = cvae.continuous_bernoulli_loss(in_sample, output['reconstruction'])
    loss2 = cvae.kl_divergence_loss(in_sample, output['kl_divergence'])
    vae_loss = loss1 + loss2    # label 1

    out_sample = in_sample
    output2 = cvae.predict(out_sample)
    loss2 = cvae.continuous_bernoulli_loss(out_sample, output2['reconstruction'])  # label 0
    vae_loss2 = loss1+loss2
    elbo = tf.reduce_logsumexp(vae_loss2 - vae_loss, axis=0)

    return elbo

    #thershold for loss etc?
def thershold(model, x_latent, x_test):
    x_mnist = model.predict(x_latent, batch_size=64) #array of data latent vectors
    test_digit = model.predict(x_test, batch_size=1)  #your testdigit latent vector
    #calc the distance
    from scipy.spatial import distance
    threshold = 0.3  #min eucledian distance
    for vector in x_mnist:
        dst = distance.euclidean(vector, test_digit[0])
        if dst <= threshold:
            return True

# call this in train.py
# pro_result = evaluating(cvae, x_val)