import collections
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

def evaluating(cvae, dataset, decoder_dist, dataset_type):

    # Evaluate on Validation data
    x_train, x_val, x_test = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type)
    cvae.evaluate(x_val)

    # Computes an importance weighted log likelihood estimate.
    origl_probs = collections.defaultdict(list)
    corrct_probs = collections.defaultdict(list)

    # Computes original log likelihood estimate.
    in_sample = next(iter(x_val))
    in_sample = np.expand_dims(in_sample, axis=0)
    cvae.num_samples = 100
    output = cvae.predict(in_sample)
    klloss = cvae.kl_divergence_loss(in_sample, output['kl_divergence'])
    if decoder_dist == "cBern":
        recoloss = cvae.continuous_bernoulli_loss(in_sample, output['reconstruction'])
    elif decoder_dist == "cat":
        recoloss = cvae.categorical_loss(in_sample, output['reconstruction'])
    elbo = recoloss - klloss
    origl_probs[dataset].append(elbo.numpy())

    #Computes correction log likelihood estimate.
    #target = in_sample[1]
    cvae.appy_correction = True
    #corrct_probs[dataset].append(elbo -correct(target).sum(axis=(1, 2, 3)))

    #origl_probs[dataset] = np.concatenate(origl_probs[dataset], axis=0)
    #corrct_probs[dataset] = np.concatenate(corrct_probs[dataset], axis=0)

    return {'orig_probs': origl_probs, 'corr_probs': corrct_probs}


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


checkpoint_epoch = '0850'
dataset_type = 'grayscale'  #'grayscale'  'natural'
dataset = 'emnist'  #'mnist'  'emnist' 'cifar10'  'svhn' 'gtsrb'
decoder_dist = 'cBern'  #'cBern'  'cat'
method = 'BC-LL-no-CS'  #'BC-LL-CS'

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

cvae.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={'reconstruction': cvae.get_reconstruction_loss_func(),
          'kl_divergence': cvae.kl_divergence_loss}
)
cvae.load_weights(f'saved_models/{decoder_dist}/{dataset_type}/{dataset}/cvae-{method}/weights-{checkpoint_epoch}')
# correction part
pro_result = evaluating(cvae, dataset, decoder_dist, dataset_type)
