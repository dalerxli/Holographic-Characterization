from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf


za_model = load_model('../models/predict_za_300.h5')
za_model.summary()

XX = za_model.input
AA = za_model.layers[-1].output
aux_model = Model(XX, AA)

print('Loading model...')
estimator = load_model('../models/predict_nalmost_400.h5')
#estimator.layers[11].trainable = False
#loss_weights = [1/(550.**2),1]
loss_weights = [1,0.01]
estimator.compile(loss='mean_squared_error', optimizer='adam', loss_weights = loss_weights)
estimator.summary()
 
# input image dimensions
img_rows, img_cols = 201,201

#unpack data
print('Unpacking data...')
with open('../datasets/data_3param.txt', 'r') as file:
    data = json.load(file)
x_train = numpy.array(data["training"][0]["train_x"])
y_train = numpy.array(data["training"][0]["train_y"])
x_test = numpy.array(data["testing"][0]["test_x"])
y_test = numpy.array(data["testing"][0]["test_y"])

#format data

print('Formatting...')
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train *= 100./255
x_test *= 100./255

#Add gaussian noise to data
def noisy(img):
    row,col,ch=input_shape
    mean=0
    var=0.05
    sigma=var**0.5
    gauss=numpy.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy

print('Adding noise...')
for img in x_train:
    img = noisy(img)

for img in x_test:
    img = noisy(img)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#z_train = y_train[:,0]
#a_train = y_train[:,1]
n_train = y_train[:,2]

#z_test = y_test[:,0]
#a_test = y_test[:,1]
n_test = y_test[:,2]

a_train = aux_model.predict(x_train)
a_test = aux_model.predict(x_test)

print('Rescaling...')

#Data normalization
def rescale(min, max, list):
    scalar=100./(max-min)
    list = (list- min)*scalar
    return list

n_train = rescale(1.38, 2.5, n_train)
n_test = rescale(1.38, 2.5, n_test)

'''
a_train = rescale(0.2, 5, a_train)
a_test = rescale(0.2, 5, a_test)


targets = list()
targets.append(y_train1)
targets.append(y_train2)
targets.append(y_train3)



testtargets = list()
testtargets.append(y_test1)
testtargets.append(y_test2)
testtargets.append(y_test3)
'''
#Fitting Parameters
batch_size = 32
epochs = 200
#estimator.summary()

estimator.fit({'image' : x_train, 'a' : a_train},
               {'n' : n_train, 'aux_n' : n_train},
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = ({'image' : x_test, 'a' : a_test},
                                 {'n' : n_test, 'aux_n' : n_test}))
#              callbacks=callbacks)

estimator.save('../models/predict_nalmost_600.h5')
