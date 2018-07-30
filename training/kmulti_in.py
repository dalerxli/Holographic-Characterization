from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Concatenate, RepeatVector
from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf


# input image dimensions
img_rows, img_cols = 201,201

#unpack data
print('Unpacking data...')
with open('../datasets/evaldata_3param.txt', 'r') as file:
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

z_train = y_train[:,0]
a_train = y_train[:,1]
n_train = y_train[:,2]

z_test = y_test[:,0]
a_test = y_test[:,1]
n_test = y_test[:,2]

print('Rescaling...')

def rescale_n(n):
    scalar = 2.5-1.38
    n = (n-1.38)*100/scalar
    return n

def rescale_a(a):
    scalar = 5.0-0.2
    a = (a-0.2)*100/scalar
    return a


for i in range(x_train.shape[0]):
    n_train[i] = rescale_n(n_train[i])
    #a_train[i] = rescale_a(a_train[i])
    
for i in range(x_test.shape[0]):
    n_test[i] = rescale_n(n_test[i])
    #a_test[i] = rescale_a(a_test[i])


targets = list()
#targets.append(z_train)
targets.append(a_train)
targets.append(n_train)


testtargets = list()
#testtargets.append(z_test)
testtargets.append(a_test)
testtargets.append(n_test)


#aux_loss = K.variable(0.2)

def multi_model():
    model_input = keras.Input(shape=input_shape, name='image')
    x = model_input
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (4,4))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    conv_out = Dense(20, activation='linear')(x)
    aux_out = Dense(1, activation='linear', name='aux_n')(conv_out)
    aux_in1 = keras.Input(shape=(1,), name='a')
    #mid_aux1 = Dense(20, init = 'ones')(aux_in1)
    mid_aux1 = RepeatVector(20)(aux_in1)
    mid_aux1 = Flatten()(mid_aux1)
    #aux_in2 = keras.Input(shape=(1,), name='z')
    x = Concatenate(axis=1)([conv_out, mid_aux1])#, aux_in2])
    #x = Dense(10, activation='relu')(x)
    #x = Dropout(0.2)(x)
    #x = Dense(100, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    #x = Dropout(0.2)(x)
    model_output = Dense(1,activation='linear', name='n')(x)
    '''
    model_output = list()
    out_names = ['a', 'n']
    for i in range(0,2):
        out_name = out_names[i]
        local_output= x
#        local_output = Dense(40, activation='relu')(local_output)
        local_output = Dense(units=1, activation='linear', name = out_name)(local_output)
        model_output.append(local_output)
    '''
    #z_precision = (0.004/0.135)**(-2)
    #a_precision = (0.001)**(-2)
    #n_precision = (0.001)**(-2)
    #loss_weights = [z_precision, a_precision, n_precision]
    #Adamlr = keras.optimizers.Adam(lr=0.001)
    model = Model(inputs=[model_input, aux_in1],# aux_in2],
                  outputs=[model_output, aux_out])
    aux_loss =0.2
    model.compile(loss = 'mean_squared_error', optimizer='adam', loss_weights=[1,aux_loss])
    model.summary()
    return model
    

#Fitting Parameters
batch_size = 32
epochs = 200
seed=7

#Callbacks
callbacks = []

tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
earlystopCB = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=2000, patience=10, verbose=0, mode='auto', baseline=None)
#modelCheck = keras.callbacks.ModelCheckpoint('../models/predict_n_aux_check.h5', monitor = 'val_n_loss', verbose=1, save_best_only=True)

class ReduceAuxLoss(Callback):
    def __init__(self, monitor='val_n_loss'):
        self.monitor = monitor
  #  def _reset(self):
  #      self.aux_loss = 0.2
#    def on_train_begin(self):
 #       self._reset()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('val_n_loss')
        if loss <= 100:
            self.aux_loss = 0.1
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

reduceAux = ReduceAuxLoss()

callbacks.append(tbCallBack)
#callbacks.append(reduceAux)
#callbacks.append(modelCheck)
#callbacks.append(ProbabilityCallback)
#callbacks.append(earlystopCB)


estimator = multi_model()
#estimator.summary()

#inputs = numpy.array([x_train, a_train])

estimator.fit({'image' : x_train, 'a' : a_train},#, 'z' : z_train},
              {'n' : n_train, 'aux_n' : n_train},
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = ({'image' : x_test, 'a' : a_test},#, 'z': z_test},
                                 {'n' : n_test, 'aux_n' : n_test}),
              callbacks=callbacks)

estimator.save('../models/predict_n_multi_auxrepeat.h5')
 
