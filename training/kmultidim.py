from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
from PIL import Image

# input image dimensions
img_rows, img_cols = 201,201

file_head = '../datasets/labdata/'

print('Opening data...')

with open(file_head + 'train/' + 'data.txt', 'r') as file:
    trainset = json.load(file)
    
img_train = []
#rx_train = []
#ry_train = []
z_train = []
r_p = trainset['r_p']
a_train = trainset['a_p']
n_train = trainset['n_p']
files = trainset['filename']
for i in range(len(files)):
    filename = file_head  + files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_train.append(im_np)
    im.close()
    #rx_train.append(r_p[i][0])
    #ry_train.append(r_p[i][1])
    z_train.append(r_p[i][2])       


img_train = numpy.array(img_train).astype('float32')
#rx_train = numpy.array(rx_train).astype('float32')
#ry_train = numpy.array(ry_train).astype('float32')
a_train = numpy.array(a_train).astype('float32')
z_train = numpy.array(z_train).astype('float32')
n_train = numpy.array(n_train).astype('float32')

with open(file_head + 'test/' + 'data.txt', 'r') as file:
    testset = json.load(file)

img_test = []
#rx_test = []
#ry_test = []
z_test = []
r_p = testset['r_p']
a_test = testset['a_p']
n_test = testset['n_p']
files = testset['filename']
for i in range(len(files)):
    filename = file_head +  files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_test.append(im_np)
    im.close()
    #rx_test.append(r_p[i][0])
    #ry_test.append(r_p[i][1])
    z_test.append(r_p[i][2])


img_test = numpy.array(img_test).astype('float32')
#rx_test = numpy.array(rx_test).astype('float32')
#ry_test = numpy.array(ry_test).astype('float32')
a_test = numpy.array(a_test).astype('float32')
z_test = numpy.array(z_test).astype('float32')
n_test = numpy.array(n_test).astype('float32')

img_train *= 1./255
img_test *= 1./255

#format data

print('Formatting...')
if K.image_data_format() == 'channels_first':
    img_train = img_train.reshape(img_train.shape[0], 1, img_rows, img_cols)
    img_test = img_test.reshape(img_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    img_train = img_train.reshape(img_train.shape[0], img_rows, img_cols, 1)
    img_test = img_test.reshape(img_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


#Data normalization
def rescale(min, max, list):
    scalar=1./(max-min)
    list = (list- min)*scalar
    return list

print('Rescaling target data...')
z_train = rescale(50, 600, z_train)
z_test = rescale(50, 600, z_test)

a_train = rescale(0.2, 5, a_train)
a_test = rescale(0.2, 5, a_test)

n_train = rescale(1.38, 2.5, n_train)
n_test = rescale(1.38, 2.5, n_test)


def multioutput_model():    
    model_input = keras.Input(shape=input_shape, name='image')
    x = model_input
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (4,4))(x)
    x = Flatten()(x)
    reg = 0.01
    x = Dense(20, activation='relu', kernel_regularizer = regularizers.l2(reg))(x)
    '''
    a_model = load_model('../models/predict_a.h5')
    XX = a_model.input
    conv_out = a_model.layers[9].output
    conv_model = Model(XX, conv_out)
    model_input = keras.Input(shape=input_shape, name='image')
    x = conv_model(model_input)
    '''
    model_outputs = list()
    out_names = ['z', 'a', 'n']
    drop_rates = [0.01, 0.01, 0.2]
    regularizer_rates = [0.3, 0.3, 0.3]
    dense_nodes = [40, 20, 100]
    loss_weights = []
    for i in range(3):
        out_name = out_names[i]
        #drop_rate = drop_rates[i]
        #reg = regularizer_rates[i]
        dense_node = dense_nodes[i]
        drop_rate = 0.001
        local_output= x
        local_output = Dense(dense_node, activation='relu', kernel_regularizer = regularizers.l2(reg))(local_output)
        local_output = Dropout(drop_rate)(local_output)
        local_output = Dense(units=1, activation='linear', name = out_name)(local_output)
        model_outputs.append(local_output)
        loss_weights.append(1)

    Adamlr = keras.optimizers.Adam(lr=0.001)
    model = Model(model_input, model_outputs)
    model.compile(loss = 'mean_squared_error', optimizer='rmsprop', loss_weights = loss_weights)
    model.summary()
    return model
    

#Fitting Parameters
batch_size = 64
epochs = 200
seed=7

#Callbacks
callbacks = []

tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
earlystopCB = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=2000, patience=10, verbose=0, mode='auto', baseline=None)
'''
class ProbabilityCallback(Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        self.alpha = self.alpha - 0.1
        self.beta = self.beta + 0.1
'''

callbacks.append(tbCallBack)
#callbacks.append(ProbabilityCallback)
#callbacks.append(earlystopCB)


estimator = multioutput_model()


estimator.fit({'image' : img_train},
              {'z' : z_train, 'a' : a_train,
               'n': n_train},
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = ({'image' : img_test},
                                 {'z' : z_test, 'a' : a_test,
                                  'n': n_test}),
              callbacks=callbacks)

estimator.save('../models/predict_lab_stamp.h5')
