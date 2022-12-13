import tensorflow as tf

from re_implementation import dataset_utils_EC
from re_implementation.helpers import model_helper

# todo training = True check
# todo bernoulli and gasussian
# todo logits clip value or give prob?

# todo instance normalization layers

"""
Notes:
1) Tensorflow 2 is not clear enough.
2) Paddings cannot be determined only from paper. They refer to Likelihood regret paper saying their network is near
identical but that paper also does not specify their network structure. One needs to check the official github repo
for that paper too.
3) They only apply de-biasing on evaluation not while training.
4) They only apply importance weighting on evaluation.
"""

# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# tf.config.run_functions_eagerly(True)

train = True
continue_ckpt = False
checkpoint_epoch = '0962'

dataset_type = 'grayscale'
# dataset_type = 'natural'

dataset = 'mnist'
# dataset = 'fmnist'

# dataset = 'cifar10'
# dataset = 'svhn'
# dataset = 'gtsrb'

# decoder_dist = 'cBern'
decoder_dist = 'cat'

epochs = 1000
batch_size = 64
latent_dimensions = 20
num_samples = 1

contrast_normalize = False

if contrast_normalize:
    method = 'BC-LL-CS'
else:
    method = 'BC-LL-no-CS'

x_train, x_val, x_test = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type, contrast_normalize)

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

if continue_ckpt:
    cvae.load_weights(f'saved_models/{decoder_dist}/{dataset_type}/{dataset}/cvae-{method}/weights-{checkpoint_epoch}')
else:
    checkpoint_epoch = 0

cvae.encoder.print_network()
cvae.decoder.print_network()
cvae.print_network()

cvae.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={'reconstruction': cvae.get_reconstruction_loss_func(),
          'kl_divergence': cvae.kl_divergence_loss}
)

if train:
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'saved_models/{decoder_dist}/{dataset_type}/{dataset}/cvae-{method}/weights-' + '{epoch:04d}',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    cvae.fit(
        x=x_train,
        y={'reconstruction': x_train, 'kl_divergence': x_train},
        validation_data=(x_val, {'reconstruction': x_val, 'kl_divergence': x_val}),
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=int(checkpoint_epoch),
        callbacks=checkpoint_cb,
        verbose=1
    )

    print("Now, you can load the best model and then evaluate on test with 'train=False' and 'continue_ckpt=True'.")

else:
    cvae.evaluate(
        x=x_test,
        y={'reconstruction': x_test, 'kl_divergence': x_test},
        batch_size=batch_size,
    )
