#from dataset_utils_EC import noise
import dataset_utils_EC as uti
import matplotlib.pyplot as plt
from matplotlib.image import imread
import bias_corrections_EC as bia
import numpy as np
import tensorflow as tf
#import train.py
from helpers import model_helper

#A = uti.noise('val','grayscale')
#A = uti.noise('val','color')

#print(A[0].shape)


#plt.imshow(A[0], cmap=plt.cm.binary) # printing 10th image. You may use cmap='gray'
#plt.colorbar() # shows the bar on the right side of the image
#plt.grid(True) # will shot the grid
#plt.show()


#print(len(output))

A, B, C = uti.load_cifar10("cBern",0.9)

#A, B, C = uti.load_mnist("cBern",0.9)

#A, B, C = uti.load_fmnist("cBern",0.9)

#A, B, C = uti.load_emnist("cBern",0.9)

#A, B, C = uti.load_svhn("cBern",0.9)

#A, B, C = uti.load_celebA(0.9)

#A, B, C = uti.load_gtrsb("cBern",0.9)

#print("PROVA", A[0].shape)

#print("PROVA_2", type(A[0]))

#print("TEST", len(A[0]), len(B[0]), len(C[0]))

#print("TEST_2", A[0].shape, B[0].shape, C[0].shape)

#print(A[0][0][0][0])

images = A[0][:2,:,:,:]

print("TEST", images.shape)

#ima= B[0][0]

ima = B[0][:1,:,:,:]
print("TEST_2", ima.shape)

test_set = B[0][:3,:,:,:]
#plt.imshow(ima)
#plt.show()

#aug =uti.augment(images, 1, width_sh_rg=0.1, height_sh_rg=0.1, hor_flp = True)

#print(image)

#for im in aug:
#    plt.imshow(ima)
#    plt.show()
#print(image)

#new = uti.contrast_normalization(image)

#plt.imshow(image, cmap=plt.cm.binary) # printing 10th image. You may use cmap='gray'
#plt.colorbar() # shows the bar on the right side of the image
#plt.grid(True) # will shot the grid
#plt.show()
#print("Class ID: %s and Class name: %s" % (ytrain[index], class_names[ytrain[index][0]]))
#print(n2.shape)

#plt.imshow(new, cmap=plt.cm.binary) # printing 10th image. You may use cmap='gray'
#plt.colorbar() # shows the bar on the right side of the image
#plt.grid(True) # will shot the grid
#plt.show()

##print("NORMALISED",new.shape)

###lamdas = np.random.rand(32,32, 1).clip(min=0.001, max= 0.999)

###correction = bia.analytical_bias_correction(lamdas)

###print(correction)

#print("PROVA". type(ima))
#r = ima.shape[0]
#c = ima.shape[1]
#nc = ima.shape[2]
#ima_2 = tf.cast(ima,tf.int32)

"""
output = [np.random.randint(low=0, high=256, size=(32, 32, 1)) for i in range(1)]
output = tf.convert_to_tensor(output, dtype=tf.float32, dtype_hint=None, name=None)

r = output.shape[0]
c = output.shape[1]
nc = output.shape[2]

print("PROVA", nc)

#ima_2 = tf.make_ndarray(ima_2)

pix_corrections = tf.ones((256, nc), dtype=tf.float32)
pix_corrections = 3 * pix_corrections
corrections = np.zeros((r, c, nc), dtype=float)"""

"""for k in range(nc):
    for i in range(256):
        indexes = np.where(output[:,:,k] == i)
        print("TEST_1",indexes)
        #corrections[indexes][k] = pix_corrections[i, k]
        print("TEST_2", (output[:,:, k] == i).numpy())
        #corrections[:,:, k][(output[:,:, k] == i).numpy()] = pix_corrections[i, k]
correction = np.mean(corrections)

print("TEST_3",correction)"""

"""for k in range(nc):
    #corrections = tf.map_fn(lambda x: pix_corrections[x,k], output[:,:,k])
    for i in range(r):
        for j in range(c):
            x = int(output[i][j][k])
            corrections[i,j,k] = pix_corrections[x][k]

correction = np.mean(corrections)

print("TEST_4",correction)"""


"""
#corrections = tf.map_fn(lambda x: pix_corrections[x,k], ima)

for k in range(nc):

    for i in range(r):
        for j in range(c):
            x = ima_2[i][j][k]
            corrections [i,j,k] = pix_corrections[x][k]

corrections = tf.convert_to_tensor(corrections)
correction = tf.math.reduce_mean(corrections)

print("correction", correction)


    #corrections = ima[:,:,k].map_fn(lambda x: pix_corrections[x,k])
    elem = ima[:,:,k]
    corrections[:,:,k] = tf.map_fn(lambda x: pix_corrections[x,k], elem)
    #corrections = tf.convert_to_tensor([pix_corrections[x,k] for x in ima_2[:,:,k]])

correction = tf.math.reduce_mean(corrections)

print(correction)

counter_B = tf.ones((3, 1), dtype=tf.float32)
print("1", counter_B)
counter_B = counter_B *3
print("2",counter_B)
counter_B = tf.convert_to_tensor([x-1 for x in counter_B if x>1])
#counter_B = tf.map(lambda x: x-1 if x>1,counter_B)
print("3",counter_B)
correction = tf.math.reduce_mean(counter_B)
print("RISULTATO",correction)"""

# load model
checkpoint_epoch = '0002'
dataset_type = 'natural'  #'grayscale'  'natural'
dataset = 'cifar10'  #'mnist'  'fmnist' 'cifar10'  'svhn' 'gtsrb'
#decoder_dist = 'cBern'
decoder_dist = 'cat'  #'cBern'  'cat'
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

#test bias_correction algorithms
corrections = []

for ima in test_set:
    ima = tf.expand_dims(ima, axis=0)
    bias_correction = bia.get_bias_corrected_lkl(cvae, ima, images)
    corrections.append(bias_correction)

print("RESULT", corrections)
