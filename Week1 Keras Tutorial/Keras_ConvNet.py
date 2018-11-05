# LOAD PACKAGES
import keras
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
img_rows, img_cols = 32, 32
batch_size = 25
epochs = 40
num_classes = 3
input_shape = (img_rows, img_cols, 3)
print('* Check input_shape : ', input_shape)

# DUE TO MAC'S RELATIVE PATH ISSUE (NOT NECESSARY IN WINDOWS)
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# LOAD DATAS FROM NUMPY FILE
with np.load(dir_path + '/prediction-challenge-02-data.npz') as fh:
    print("* Check what's inside in fh : ", fh.files)
    data_x = fh['data_x']
    data_y = fh['data_y']
    test_x = fh['test_x']
    print('* Check the shape of data_x : ', data_x.shape)

    # USUALLY IN KERAS, TENSORFLOW, IMAGE CHANNELS ARE THE LAST ONE
    data_x = np.transpose(data_x, (0, 2,3,1))
    test_x = np.transpose(test_x, (0, 2,3,1))
    print('* After transpose : ', data_x.shape)

    # TO SHUFFLE THE DATA (NOT NECESSARY)
    number_datapoints = data_x.shape[0]
    indexes = np.arange(number_datapoints)
    np.random.shuffle(indexes)

    length_train = int(number_datapoints*0.9)
    length_vali = number_datapoints - length_train

# CREATE DATASET WITH SEPERATED, SHUFFLED INDEXES
    x_train = data_x[indexes[:length_train]]
    y_train = data_y[indexes[:length_train]]
    x_test = data_x[indexes[length_train:]]
    y_test = data_y[indexes[length_train:]]
    x_predict = test_x # ACTUALLY THIS IS  CONSIDERED AS NULL

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 3)

    # ONE HOT VECTOR ENCODING
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("* x_train data shape : ", x_train.shape)
    print("* y_train data shape : ", y_train.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from keras import Input

# CONSISTING MODEL WITH model.add (FOR SIMPLE MODEL)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

# CONSISTING MODEL WITH x = layer(..) (FOR COMPLICATED MODEL)
'''
input = keras.Input(shape=input_shape)
x = Conv2D(32, 3, padding='same', activation='relu')(input)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(input, x)
'''

# SET TRAINING METHOD, LOSS
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINING THE MODEL (SIMPLE)
hist = model.fit(x_train, y_train,
                 epochs=50,
                 batch_size=100,
                 validation_data=(x_test, y_test)
                 )

# TRAINING THE MODEL (USING CALLBACKS)
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint
# EARLY STOPPING
early_stopping = EarlyStopping(monitor='val_acc', verbose=1)
# MAKING CHECKPOINT
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

hist = model.fit(x_train, y_train,
                 epochs=5,
                 batch_size=100,
                 shuffle=True,
                 validation_data=(x_test, y_test),
                 callbacks = [early_stopping, checkpoint]
                 )
'''

# SAVE AND LOAD MODEL (FOR CODING)
from keras.models import load_model
filename = 'Keras_ConvNet.h5'
model.save(filename)
model = load_model(filename)

# EVALUATE THE MODEL WITH VALIDATION DATA
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)
print('\n## evaluation loss and_metrics ##')
print(loss_and_metrics)

# SEE THE TRAINING HISTORY
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
fig = plt.gcf()
fig.savefig('plt.png')

# DO THE PREDICTION
xhat = x_test
yhat = model.predict(xhat)
prediction = yhat
yhat_submit = np.zeros(length_vali)
print('* prediction.shape : ', prediction.shape)
print('* yhat_submit.shape : ', yhat_submit.shape)

# REDUCE THE DIMENSION OF YHAT AS REQURIED FORMAT (DUE TO THE ONE-HOT ENCODING)
for i in range(0, length_vali) :
    maxyhat = np.max(yhat[i])
    indexyhat = np.where(yhat[i] == maxyhat)
    yhat_submit[i] = indexyhat[0]

# SAVE PREDICTION
np.save('prediction.npy', yhat_submit)

# TO SEE IMAGES
'''
from matplotlib import pyplot as plt
plt.imshow(x_test[0], interpolation='nearest')
plt.show()
print(yhat_submit[0])

plt.imshow(x_train[0], interpolation='nearest')
plt.show()
print(y_train[0])
'''
