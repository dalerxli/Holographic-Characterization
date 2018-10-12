from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
from PIL import Image

# input image dimensions
img_rows, img_cols = 480, 640


file_head = '../datasets/xydata/'

print('Opening data...')

with open(file_head + 'train/' + 'data.txt', 'r') as file:
    trainset = json.load(file)


img_train = []
rx_train = []
ry_train = []
r_p = trainset['r_p']
files = trainset['filename']
for i in range(len(files)):
    filename = file_head  + files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_train.append(im_np)
    im.close()
    rx_train.append(r_p[i][0])
    ry_train.append(r_p[i][1])

    
img_train = numpy.array(img_train).astype('float32')
img_train*= 1./255
rx_train = numpy.array(rx_train).astype('float32')
ry_train = numpy.array(ry_train).astype('float32')


with open(file_head + 'test/' + 'data.txt', 'r') as file:
    testset = json.load(file)

img_test = []
rx_test = []
ry_test = []
r_p = testset['r_p']
files = testset['filename']
for i in range(len(files)):
    filename = file_head +  files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_test.append(im_np)
    im.close()
    rx_test.append(r_p[i][0])
    ry_test.append(r_p[i][1])

    
img_test = numpy.array(img_test).astype('float32')
img_test *= 1./255
rx_test = numpy.array(rx_test).astype('float32')
ry_test = numpy.array(ry_test).astype('float32')

'''
xround_train = numpy.around(rx_train)
yround_train = numpy.around(ry_train)

target

yround_test = numpy.around(rx_test)
yround_test = numpy.around(ry_test)
'''
print('Formatting...')
if K.image_data_format() == 'channels_first':
    img_train = img_train.reshape(img_train.shape[0], 1, img_rows, img_cols)
    img_test = img_test.reshape(img_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    img_train = img_train.reshape(img_train.shape[0], img_rows, img_cols, 1)
    img_test = img_test.reshape(img_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

def rescale(min, max, list):
    scalar = 1./(max-min)
    list = (list-min)*scalar
    return list

tol = 100
xmax = img_cols/2 - tol
xmin = -xmax
ymax = img_rows/2 - tol
ymin = -ymax

rx_train = rescale(xmin, xmax, rx_train)
rx_test = rescale(xmin, xmax, rx_test)
ry_train = rescale(ymin, ymax, ry_train)
ry_test = rescale(ymin, ymax, ry_test)

print(numpy.amin(rx_train), 'min')
print(numpy.amax(rx_train), 'max')

def create_model():
    model_input = keras.Input(shape=input_shape, name='image')
    x = model_input
    x = AveragePooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(5, 5), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation = 'relu')(x)
    x = AveragePooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation = 'relu')(x)
    x = AveragePooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (4,4))(x)
    x = Flatten()(x)
    model_outputs = list()
    out_names = ['x', 'y']
    for i in range(2):
        out_name = out_names[i]
        local_output= x
        local_output = Dense(10, activation='relu')(local_output)
        local_output = Dropout(0.001)(local_output)
        local_output = Dense(units=1,  name = out_name)(local_output)
        model_outputs.append(local_output)
    model = Model(input=model_input, output= model_outputs)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


#Fitting Parameters
batch_size = 64
epochs = 20
seed=7

#Callbacks
callbacks = []

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
earlystopCB = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=2000, patience=10, verbose=0, mode='auto', baseline=None)
callbacks.append(tbCallBack)
#callbacks.append(earlystopCB)


estimator = create_model()

estimator.summary()

estimator.fit({'image': img_train}, {'x' : rx_train, 'y': ry_train},
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = ({'image': img_test}, {'x' : rx_test, 'y' : ry_test}),
              callbacks=callbacks)

#bias = estimator.layers[-1].get_weights()[1]
#scores = estimator.evaluate(x=x_test, y=y_test, verbose=1)
#print("Final bias", bias)
#print("MSE", scores)
estimator.save('../models/predict_xy4.h5')
