# Maxpooling2D(), UpSampling2D() layers should be replaced to transfer indices

from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import json

img_w = 960
img_h = 720
n_labels = 32

kernel = 3

encoding_layers = [
    Conv2D(64, kernel, border_mode='same', input_shape=( img_h, img_w,1)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    #MaxPooling2D(),
]

autoencoder = models.Sequential()
autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)

decoding_layers = [
    #UpSampling2D(),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]

autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Reshape((n_labels, img_h * img_w)))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))

autoencoder.summary()

# SAVE AND LOAD MODEL (FOR CODING)
from keras.models import load_model
import os
filename = os.path.join(os.path.dirname((os.path.realpath(__file__))), 'SegNet.h5')
print(filename)
autoencoder.save(filename)
