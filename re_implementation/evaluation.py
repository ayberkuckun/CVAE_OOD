import numpy as np
import sklearn.metrics
import tensorflow as tf
from tqdm import tqdm

from re_implementation import dataset_utils_EC, bias_corrections_EC
from re_implementation.helpers import model_helper


# tf.keras.mixed_precision.set_global_policy('mixed_float16')

dataset_type = 'grayscale'
# dataset_type = 'natural'

if dataset_type == "grayscale":
    dataset_list = [
        "mnist",
        "fmnist",
        "emnist",
        "noise"
    ]
elif dataset_type == "natural":
    dataset_list = [
        "cifar10",
        "svhn",
        "gtsrb",
        "noise"
    ]
else:
    raise ValueError

dataset = 'mnist'
# dataset = 'emnist'

# dataset = 'cifar10'
# dataset = 'svhn'
# dataset = 'gtsrb'

# decoder_dist = 'cBern'
decoder_dist = 'cat'

# checkpoint_epoch = '0878'
checkpoint_epoch = '0192'
# checkpoint_epoch = '0039'
# checkpoint_epoch = '0011'
# checkpoint_epoch = '0870'
# checkpoint_epoch = '0858'

latent_dimensions = 20
num_samples = 1

normalization = "batch"
# normalization = "instance"

contrast_normalize = True

if contrast_normalize:
    method = f'BC-LL-CS-{normalization}'
else:
    method = f'BC-LL-no-CS-{normalization}'

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

# ------------------------ Evaluation --------------------------------- #
cvae.num_samples = 1
cvae.apply_mean = False
contrast_normalize = True

x_train, _, x_test_id = dataset_utils_EC.get_dataset(dataset, decoder_dist, dataset_type, contrast_normalize, training=False)
x_test_id_batched = tf.split(x_test_id, 100 * cvae.num_samples)

if decoder_dist == "cat":
    pix_corrections = bias_corrections_EC.algorithmic_bias_correction(cvae, x_train)
    correction_func = None
else:
    pix_corrections = None
    correction_func = bias_corrections_EC.get_correction_func()

ll_list = []
cr_list = []
for x_test_batch in tqdm(x_test_id_batched):
    ll = cvae.likelihood(x_test_batch, training=False)
    ll_list.append(ll)

    cr = bias_corrections_EC.get_bias_corrected_lkl(cvae.decoder_dist, x_test_batch, correction_func, pix_corrections)
    cr_list.append(cr)

ll_id = tf.concat(ll_list, axis=0)
cr_id = tf.concat(cr_list, axis=0)
bc_id = ll_id - cr_id  # plus or minus?

auroc_list_ll = []
auroc_list_bc_ll = []
for dataset_ood in dataset_list:
    _, _, x_test_ood = dataset_utils_EC.get_dataset(dataset_ood, decoder_dist, dataset_type, contrast_normalize, training=False)
    x_test_ood_batched = tf.split(x_test_ood, 100 * cvae.num_samples)

    ll_list = []
    cr_list = []
    for x_test_batch in tqdm(x_test_ood_batched):
        ll = cvae.likelihood(x_test_batch, training=False)
        ll_list.append(ll)

        cr = bias_corrections_EC.get_bias_corrected_lkl(cvae.decoder_dist, x_test_batch, correction_func, pix_corrections)
        cr_list.append(cr)

    ll_ood = tf.concat(ll_list, axis=0)
    cr_ood = tf.concat(cr_list, axis=0)
    bc_ood = ll_id - cr_ood  # plus or minus?

    # --- LL --- #

    y_true = np.concatenate([np.zeros_like(ll_ood), np.ones_like(ll_id)])
    y_score = np.concatenate([ll_ood, ll_id])

    auroc_ll = sklearn.metrics.roc_auc_score(y_true, y_score)
    auroc_list_ll.append(auroc_ll)

    print(f"{dataset}_VAE/{dataset_ood}-LL-AUROC: {auroc_ll}")

    # --- BC-LL --- #

    y_true = np.concatenate([np.zeros_like(bc_ood), np.ones_like(bc_id)])
    y_score = np.concatenate([bc_ood, bc_id])

    auroc_bc_ll = sklearn.metrics.roc_auc_score(y_true, y_score)
    auroc_list_bc_ll.append(auroc_bc_ll)

    print(f"{dataset}_VAE/{dataset_ood}-BC-LL-AUROC: {auroc_bc_ll}")


for no, ll in enumerate(zip(auroc_list_ll, auroc_list_bc_ll)):
    auroc_ll, auroc_bc_ll = ll
    print(f"{dataset}_VAE/{dataset_list[no]}-LL-AUROC: {auroc_ll}")
    print(f"{dataset}_VAE/{dataset_list[no]}-BC-LL-AUROC: {auroc_bc_ll}")

print(f"{dataset}_VAE/Average-LL-AUROC: {np.mean(auroc_list_ll)}")
print(f"{dataset}_VAE/Average-BC-LL-AUROC: {np.mean(auroc_list_bc_ll)}")
