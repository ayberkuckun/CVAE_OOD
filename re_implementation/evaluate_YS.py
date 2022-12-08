

import numpy as np
from scipy.optimize import minimize
from numpy.random import rand
import tensorflow as tf
import tensorflow_datasets as tfds
import itertools
from absl import flags


_NORMALIZE = flags.DEFINE_string(
    'normalize', default=None, help='Normalization to apply')

dataset = 'mnist'
batch_size = 64

model_func = {
    'metrics': [
        tf.keras.metrics.Accuracy(),
    ]
}

def getValData(dataset):
    if dataset == "mnist":
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist", split=['train[:90%]', 'train[90%:]', 'test'], with_info=True)

        ds_val = ds_val.map(
            partial(preprocess, inverted=False, mode='grayscale',
                    normalize=_NORMALIZE.value, dequantize=False,
                    visible_dist='cont_bernoulli'),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.cache()
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_val

    else:
        return None


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
def evaluating(model, dataset):
    if dataset == "mnist":
        ds_val = getValData(dataset)
        # model.evaluate(ds_val)
        model.evaluate_generator(generator=validation_generator,
                                 workers=10,
                                 use_multiprocessing=True,
                                 verbose=0)
        return model
    else:
        return None



model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={'reconstruction': cvae.get_reconstruction_loss_func(),
          'kl_divergence': cvae.kl_divergence_loss},
    metrics=model_func["metrics"]
)