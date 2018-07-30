from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf


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

y_train1 = y_train[:,0]
y_train2 = y_train[:,1]
y_train3 = y_train[:,2]


targets = list()
targets.append(y_train1)
targets.append(y_train2)
targets.append(y_train3)


y_test1 = y_test[:,0]
y_test2 = y_test[:,1]
y_test3 = y_test[:,2]

testtargets = list()
testtargets.append(y_test1)
testtargets.append(y_test2)
testtargets.append(y_test3)


def sequential_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
 #   model.add(Dense(15))
#    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    #Adamlr = keras.optimizers.Adam(lr=0.0007)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def multioutput_model():
    model_input = keras.Input(shape=input_shape)
    x = model_input
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (4,4))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
#    x = Reshape((800,1))(x)
#    x = LSTM(15, input_shape = (800,1), return_sequences=False)(x)
    x = Dense(4, activation='relu')(x)
#    x = Dropout(0.2)(x)
#    x = Dense(20, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    model_outputs = list()
    out_names = ['z', 'n', 'a']
    for i in range(0,3):
        out_name = out_names[i]
        local_output= x
        if i ==2:
            local_output = Dense(50, activation='relu')(local_output)
        local_output = Dense(units=1, activation='linear', name = out_name)(local_output)
        model_outputs.append(local_output)
        
    loss_weights = [1,10000,10000]
    Adamlr = keras.optimizers.Adam(lr=0.01)
    model = Model(model_input, model_outputs)
    model.compile(loss = 'mean_squared_error', optimizer=Adamlr, loss_weights = loss_weights)
    model.summary()
    return model
    

#Fitting Parameters
batch_size = 100
epochs = 200
seed=7

#Callbacks
callbacks = []

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
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


estimator = sequential_model()
estimator.summary()

estimator.fit(x_train, y_train2,
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = (x_test, y_test2),
              callbacks=callbacks)

#scores = estimator.evaluate(x=x_test, y=testtargets, verbose=1)
#print("MSE", scores)
estimator.save('../models/predict_a.h5')
'''
#estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
print('Predicted:',y_pred[0])
print('Actual:',y_test[0])

#plt.plot(y_test, y_pred, 'ro')
#plt.plot(y_test, y_test)
#plt.show()

'''
#estimator_json = estimator.model.to_json()
#print(estimator_json)
#with open('json_model.txt', 'w') as outfile:
#        json.dump(estimator_json, outfile)
