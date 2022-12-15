import numpy as np
import sklearn
import tensorflow as tf

from re_implementation import dataset_utils_EC
from re_implementation.helpers import model_helper


# loss smaller with clipping?
################## OOD Dataset ##################
dataset_type = 'grayscale'
# dataset_type = 'natural'

dataset = 'mnist'
# dataset = 'emnist'

# dataset = 'cifar10'
# dataset = 'svhn'
# dataset = 'gtsrb'

decoder_dist = 'cBern'
# decoder_dist = 'cat'

contrast_normalize = True

_, _, x_test_ood = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type, contrast_normalize)

################## Model ##################
checkpoint_epoch = '0962'

dataset_type = 'grayscale'
# dataset_type = 'natural'

dataset = 'mnist'
# dataset = 'emnist'

# dataset = 'cifar10'
# dataset = 'svhn'
# dataset = 'gtsrb'

decoder_dist = 'cBern'
# decoder_dist = 'cat'

latent_dimensions = 20
num_samples = 100

normalization = "batch"
# normalization = "instance"

contrast_normalize = True

if contrast_normalize:
    method = f'BC-LL-CS-{normalization}'
else:
    method = f'BC-LL-no-CS-{normalization}'

_, _, x_test_id = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type, contrast_normalize)

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
    decoder_dist=decoder_dist,
    normalization=normalization
)

cvae.load_weights(f'saved_models/{decoder_dist}/{dataset_type}/{dataset}/cvae-{method}/weights-{checkpoint_epoch}')

cvae.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={'reconstruction': cvae.get_reconstruction_loss_func(),
          'kl_divergence': cvae.kl_divergence_loss}
)

ll_ood = tf.map_fn(
    lambda x: cvae.evaluate(
        x=x,
        y={'reconstruction': x, 'kl_divergence': x},
    ),
    x_test_ood,
    parallel_iterations=20,
    back_prop=False,
)

cvae.appy_correction = True

bc_ll_ood = tf.map_fn(
    lambda x: cvae.evaluate(
        x=x,
        y={'reconstruction': x, 'kl_divergence': x},
    ),
    x_test_ood,
    parallel_iterations=20,
    back_prop=False,
)

ll_id = tf.map_fn(
    lambda x: cvae.evaluate(
        x=x,
        y={'reconstruction': x, 'kl_divergence': x},
    ),
    x_test_id,
    parallel_iterations=20,
    back_prop=False,
)

cvae.appy_correction = True

bc_ll_id = tf.map_fn(
    lambda x: cvae.evaluate(
        x=x,
        y={'reconstruction': x, 'kl_divergence': x},
    ),
    x_test_id,
    parallel_iterations=20,
    back_prop=False,
)

y_true = np.concatenate([np.zeros_like(ll_ood),
                          np.ones_like(ll_id)])

y_score = np.concatenate([ll_ood,
                      ll_id])

auroc = sklearn.metrics.roc_auc_score(y_true, y_score)

print(f"LL - AUROC: {auroc}")

y_true_bc = np.concatenate([np.zeros_like(bc_ll_ood),
                          np.ones_like(bc_ll_id)])

y_score_bc = np.concatenate([bc_ll_ood,
                      bc_ll_id])

auroc = sklearn.metrics.roc_auc_score(y_true_bc, y_score_bc)

print(f"BC-LL AUROC: {auroc}")
