import tensorflow as tf

from re_implementation import dataset_utils_EC
from re_implementation.helpers import model_helper

checkpoint_epoch = '0962'

dataset_type = 'grayscale'
# dataset_type = 'natural'

dataset = 'mnist'
# dataset = 'emnist'

# dataset = 'cifar10'
# dataset = 'svhn'
# dataset = 'gtsrb'

# decoder_dist = 'cBern'
decoder_dist = 'cat'

latent_dimensions = 20
num_samples = 100

normalization = "batch"
# normalization = "instance"

contrast_normalize = True

if contrast_normalize:
    method = f'BC-LL-CS-{normalization}'
else:
    method = f'BC-LL-no-CS-{normalization}'

_, _, x_test = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type, contrast_normalize)

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

tf.map_fn(
    lambda x: cvae.evaluate(
        x=x,
        y={'reconstruction': x, 'kl_divergence': x},
    ),
    x_test,
    parallel_iterations=20,
    back_prop=False,
)

x_out = cvae(x_test, training=False)

image = x_test[0]
out_image = x_out["reconstruction"][0]

rec_loss = cvae.get_reconstruction_loss_func()(image, out_image)
kl_loss = cvae.get_reconstruction_loss_func()(image, out_image)



cvae.appy_correction = True