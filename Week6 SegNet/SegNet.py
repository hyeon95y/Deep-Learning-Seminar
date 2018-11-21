import numpy as np
import pandas as pd

import json
import sys

from skimage.io import imread
from matplotlib import pyplot as plt

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

from keras import models
from keras.optimizers import SGD


path = os.path.join(os.path.dirname(__file__), 'CamSeq01/')
print(path)
img_w = 960
img_h = 720
n_labels = 32

n_train = 82
n_test = 19


def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])
    for r in range(img_h):
        for c in range(img_w):
            print(r, c, labels[r][c])
            #label_map[r, c, labels[r][c]] = 1

            if np.array_equal(labels[r][c], [64, 128, 64]) :
                label_map[r,c,0] = 1
            elif np.array_equal(labels[r][c], [192, 0, 128]) :
                label_map[r, c, 1] = 1
            elif np.array_equal(labels[r][c], [0, 128, 192]) :
                label_map[r, c, 2] = 1
            elif np.array_equal(labels[r][c], [0, 128, 64]) :
                label_map[r, c, 3] = 1
            elif np.array_equal(labels[r][c], [128, 0, 0]) :
                label_map[r, c, 4] = 1
            elif np.array_equal(labels[r][c], [64, 0, 128]) :
                label_map[r, c, 5] = 1
            elif np.array_equal(labels[r][c], [64, 0, 192]) :
                label_map[r, c, 6] = 1
            elif np.array_equal(labels[r][c], [192, 128, 64]) :
                label_map[r, c, 7] = 1
            elif np.array_equal(labels[r][c], [192, 192, 128]) :
                label_map[r, c, 8] = 1
            elif np.array_equal(labels[r][c], [64, 64, 128]) :
                label_map[r, c, 9] = 1
            elif np.array_equal(labels[r][c], [128, 0, 192]) :
                label_map[r, c, 10] = 1
            elif np.array_equal(labels[r][c], [192, 0, 64]) :
                label_map[r, c, 11] = 1
            elif np.array_equal(labels[r][c], [128, 128, 64]) :
                label_map[r, c, 12] = 1
            elif np.array_equal(labels[r][c], [192, 0, 192]) :
                label_map[r, c, 13] = 1
            elif np.array_equal(labels[r][c], [128, 64, 64]) :
                label_map[r, c, 14] = 1
            elif np.array_equal(labels[r][c], [64, 192, 128]) :
                label_map[r, c, 15] = 1
            elif np.array_equal(labels[r][c], [64, 64, 0]) :
                label_map[r, c, 16] = 1
            elif np.array_equal(labels[r][c], [128, 64, 128]) :
                label_map[r, c, 17] = 1
            elif np.array_equal(labels[r][c], [128, 128, 192]) :
                label_map[r, c, 18] = 1
            elif np.array_equal(labels[r][c], [0, 0, 192]) :
                label_map[r, c, 19] = 1
            elif np.array_equal(labels[r][c], [192, 128, 128]) :
                label_map[r, c, 20] = 1
            elif np.array_equal(labels[r][c], [128, 128, 128]) :
                label_map[r, c, 21] = 1
            elif np.array_equal(labels[r][c], [64, 128, 192]) :
                label_map[r, c, 22] = 1
            elif np.array_equal(labels[r][c], [0, 0, 64]) :
                label_map[r, c, 23] = 1
            elif np.array_equal(labels[r][c], [0, 64, 64]) :
                label_map[r, c, 24] = 1
            elif np.array_equal(labels[r][c], [192, 64, 128]) :
                label_map[r, c, 25] = 1
            elif np.array_equal(labels[r][c], [128, 128, 0]) :
                label_map[r, c, 26] = 1
            elif np.array_equal(labels[r][c], [192, 128, 192]) :
                label_map[r, c, 27] = 1
            elif np.array_equal(labels[r][c], [64, 0, 64]) :
                label_map[r, c, 28] = 1
            elif np.array_equal(labels[r][c], [192, 192, 0]) :
                label_map[r, c, 29] = 1
            elif np.array_equal(labels[r][c], [0, 0, 0]) :
                label_map[r, c, 30] = 1
            elif np.array_equal(labels[r][c], [64, 192, 0]) :
                label_map[r, c, 31] = 1




    return label_map

import glob
from PIL import Image

def prep_data(mode):
    if mode == 'train' :
        n = n_train
    elif mode == 'test' :
        n = n_test

    data = []
    label = []
    #df = pd.read_csv(path +  mode + '.csv')
    filelist = [
        glob.glob(os.path.join(path, mode, "Original", "*.png")),
        glob.glob(os.path.join(path, mode, "GT", "*.png"))
    ]
    print(filelist[0])
    print(filelist[1])

    data = np.array([np.array(Image.open(fname)) for fname in filelist[0]])
    print(data[0].shape)
    label = np.array([label_map(np.array(Image.open(fname))) for fname in filelist[1]])




    '''
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(img)
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    '''

    #label = np.array(label).reshape((n, img_h * img_w, n_labels))
    data = data.reshape(data.shape[0], img_h, img_w, 3)

    print(mode + ': OK')
    print('\tshapes: {}, {}'.format(data.shape, label.shape))
    print('\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print('\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label


def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.png')
    plt.show()


#########################################################################################################

from keras.models import load_model
filename = os.path.join(os.path.dirname(__file__), 'SegNet.h5')
autoencoder = load_model(filename)

#with open(os.path.join(os.path.dirname(__file__), 'model_5l.json')) as model_file:
#    autoencoder = models.model_from_json(model_file.read())

optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print('Compiled: OK')

# Train model or load weights
train_data, train_label = prep_data('train')
nb_epoch = 30
batch_size = 18
history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
# autoencoder.save_weights(os.path.join(os.path.dirname(__file__), 'model_5l_weight_ep30.hdf5'))
# autoencoder.load_weights(os.path.join(os.path.dirname(__file__), 'model_5l_weight_ep30.hdf5'))
autoencoder.save(os.path.join(os.path.dirname(__file), 'SegNet(epochs30).h5'))

# Model visualization
from keras.utils.vis_utils import plot_model
plot_model(autoencoder, to_file=os.path.join(os.path.dirname(__file__), 'model.png'), show_shapes=True)
test_data, test_label = prep_data('test')
score = autoencoder.evaluate(test_data, test_label, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

output = autoencoder.predict_proba(test_data, verbose=0)
output = output.reshape((output.shape[0], img_h, img_w, n_labels))

plot_results(output)
