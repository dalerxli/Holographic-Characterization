import numpy
try:
    import h5py
except ImportError:
    h5py = None
from keras.models import load_model
#from keras import load_weights
import json
from keras import backend as K
from keras.models import Model
from matplotlib import pyplot as plt

with open('../datasets/evaldata_3param.txt', 'r') as file:
    data = json.load(file)
x_train = numpy.array(data["training"][0]["train_x"][0:100])
y_train = numpy.array(data["training"][0]["train_y"][0:100])
x_test = numpy.array(data["testing"][0]["test_x"][0:1])
y_test = numpy.array(data["testing"][0]["test_y"][0:1])

#format data

img_rows = 201
img_cols = 201
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
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

#print(x_test[0][:][0].shape)

#Add gaussian noise to data                                                                                            
def noisy(img):
    row,col,ch=input_shape
    mean=0
    var=10**(-5)
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

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

z_train = y_train[:,0]
a_train = y_train[:,1]
n_train = y_train[:,2]

z_test = y_test[:,0]
a_test = y_test[:,1]
n_test = y_test[:,2]

targets = list()
targets.append(z_train)
targets.append(n_train)
targets.append(a_train)


model = load_model('../models/predict_n_multi_aux.h5')
model.summary()

#K.set_value(model.layers[-1].weights[1], numpy.array([bias[0]+10]))

n_pred = model.predict({'image' : x_train, 'a' : a_train})[0]

def rescale_n(n):
    scalar = 2.5-1.38
    n = n*scalar/100 + 1.38
    return n

def rescale_a(a):
    scalar = 5.0-0.2
    a = a*scalar/100 + 0.2
    return a
    
for i in range(x_train.shape[0]):
    #a_pred[i] = rescale_a(a_pred[i])
    n_pred[i] = rescale_n(n_pred[i])

diff = numpy.zeros(100)
for i in range(0, 100):
    diff[i] = numpy.abs(n_train[i] - n_pred[i])
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = (diff[i])**2
SST = numpy.sum(sqer)
SST = SST/100
RMSE = numpy.sqrt(SST)
print('RMSE', RMSE)

plt.plot(n_train, diff, 'bo')
plt.show()
#print(rmsd)
#history = model.fit(x_train, y_train, epochs=100)
#plt.plot(history.history['loss'])
#plt.show()

'''
plt.plot(z_test, z_pred, 'bo')
plt.plot(z_test, z_test, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual z-value (pixels)')
plt.ylabel('Predicted z-value (pixels)')
plt.show()
'''

plt.plot(n_train, n_pred, 'bo')
plt.plot(n_train, n_train, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual a_p value (microns)')
plt.ylabel('Predicted a_p value (microns)')
plt.show()
'''
plt.plot(n_train, n_pred, 'bo')
plt.plot(n_train, n_train, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual n_p value')
plt.ylabel('Predicted n_p value')
plt.show()
'''
