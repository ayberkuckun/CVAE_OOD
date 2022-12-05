import tensorflow as tf

from reimplementation.helpers import model_helper, dataset_helper

# todo training = True check

# todo bernoulli and gasussian
# todo logits clip value or give prob?

# todo apply debiasing on training?
# todo iw apply on train?

"""
Notes:
1) Tensorflow 2 is not clear enough.
2) Paddings cannot be determined only from paper. They refer to Likelihood regret paper saying their network is near
identical but that paper also does not specify their network structure. One needs to check the official github repo
for that paper too.
3) They only apply de-biasing on evaluation not while training.
4) They only apply importance weighting on evaluation.
"""

# tf.config.run_functions_eagerly(True)

dataset_type = 'grayscale'
dataset = 'mnist'
decoder_dist = 'cBern'
method = 'BC-LL'
val_split = 0.1
epochs = 1000
batch_size = 64
latent_dimensions = 20
num_samples = 1

x_train = dataset_helper.get_dataset(dataset, decoder_dist)

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

cvae.encoder.print_network()
cvae.decoder.print_network()
cvae.print_network()

cvae.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={'reconstruction': cvae.get_reconstruction_loss_func(),
          'kl_divergence': cvae.kl_divergence_loss}
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'saved_models/{decoder_dist}/{dataset}/cvae-{method}',
    monitor='val_loss',
    save_best_only=True
)

cvae.fit(
    x=x_train,
    y={
        'reconstruction': x_train,
        'kl_divergence': x_train
    },
    batch_size=batch_size,
    epochs=epochs,
    callbacks=checkpoint_cb,
    validation_split=val_split
)
