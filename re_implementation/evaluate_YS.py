
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from re_implementation import dataset_utils_EC
from re_implementation.helpers import model_helper


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
    elbo = elbo - tf.math.log(tf.cast(100, dtype=tf.float32))

    return elbo

#thershold for loss etc?
def thershold(model, x_latent, x_test):
    x_mnist = model.predict(x_latent, batch_size=64)  #array of data  vectors
    test_digit = model.predict(x_test, batch_size=1)  #your testdigit  vector
    #calc the distance
    from scipy.spatial import distance
    threshold = 0.3  #min eucledian distance
    for vector in x_mnist:
        dst = distance.euclidean(vector, test_digit[0])
        if dst <= threshold:
            return True


checkpoint_epoch = '0004'
dataset_type = 'grayscale'  #'grayscale'  'natural'
dataset = 'cifar10'  #'mnist'  'fmnist' 'cifar10'  'svhn' 'gtsrb'
decoder_dist = 'cBern'  #'cBern'  'cat'
method = 'BC-LL'

latent_dimensions = 20
num_samples = 1

if dataset_type == 'grayscale':
    num_filter = 32
    num_channel = 1

elif dataset_type == 'natural':
    num_filter = 64
    num_channel = 3
else:
    raise ValueError('Undefined dataset type.')

cvae = model_helper.CVAE(
    num_channel=num_channel,
    num_filter=num_filter,
    latent_dimensions=latent_dimensions,
    num_samples=num_samples,
    decoder_dist=decoder_dist
)

cvae.load_weights(f'saved_models/{decoder_dist}/{dataset}/cvae-{method}/weights-{checkpoint_epoch}')
x_train, x_val, _ = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type)
pro_result = evaluating(cvae, x_val)
