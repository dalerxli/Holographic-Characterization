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
from PIL import Image

file_head = '../datasets/labdata/'

print('Opening data...')


with open(file_head + 'eval/' + 'data.txt', 'r') as file:
        evalset = json.load(file)


img_eval = []
rx_eval = []
ry_eval = []
z_eval = []
a_eval = []
r_cyl = []
r_p = evalset['r_p']
files = evalset['filename']
a_eval = evalset['a_p']
n_eval = evalset['n_p']
for i in range(len(files)):
    filename = file_head  + files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_eval.append(im_np)
    im.close()
    rx_eval.append(r_p[i][0])
    ry_eval.append(r_p[i][1])
    z_eval.append(r_p[i][2])
    r_cyl_local = r_p[i][0]**2 + r_p[i][1]**2
    r_cyl.append(r_cyl_local)


img_eval = numpy.array(img_eval).astype('float32')
img_eval*= 1./255
z_eval = numpy.array(z_eval).astype('float32')
a_eval = numpy.array(a_eval).astype('float32')
n_eval = numpy.array(n_eval).astype('float32')



img_rows = 201
img_cols = 201

if K.image_data_format() == 'channels_first':
    img_eval = img_eval.reshape(img_eval.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    img_eval = img_eval.reshape(img_eval.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



stamp_model = load_model('../models/predict_lab_stamp_zextra.h5')
#aux2 = load_model('../models/rescale_output.h5')
stamp_model.summary()

y_pred = stamp_model.predict(img_eval)
z_pred = y_pred[0]
a_pred = y_pred[1]
n_pred = y_pred[2]


def rescale_back(min, max, list):
    scalar = (max-min)/1.
    list = list*scalar +min
    return list

def rescale(min, max, list):
    scalar = 1./(max-min)
    list = (list-min)*scalar
    return list


z_pred = rescale_back(50, 600, z_pred)
a_pred  = rescale_back(0.2, 5, a_pred)
n_pred = rescale_back(1.38, 2.5, n_pred) 

diffz = numpy.zeros(100)
for i in range(0, 100):
    diffz[i] = z_eval[i] - z_pred[i]
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = (diffz[i])**2
SST = numpy.sum(sqer)
SST = SST/100
z_RMSE = numpy.sqrt(SST)
#print('z RMSE', RMSE)

plt.plot(z_eval, z_pred, 'bo')
plt.plot(z_eval, z_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual z-value (pixels)')
plt.ylabel('Predicted z-value (pixels)')
plt.text(80,500,'RMSE=%s'%z_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()

plt.plot(a_eval, numpy.abs(diffz), 'bo')
plt.title('Radius vs z error')
plt.show()



diffa = numpy.zeros(100)
for i in range(0, 100):
    diffa[i] = a_eval[i] - a_pred[i]
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = diffa[i]**2
SST = numpy.sum(sqer)
SST = SST/100
a_RMSE = numpy.sqrt(SST)

plt.plot(a_eval, a_pred, 'bo')
plt.plot(a_eval, a_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual a_p value (microns)')
plt.ylabel('Predicted a_p value (microns)')
plt.text(0.4,4,'RMSE=%s'%a_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()


diffn = numpy.zeros(100)
for i in range(0, 100):
    diffn[i] = n_eval[i] - n_pred[i]
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = diffn[i]**2
SST = numpy.sum(sqer)
SST = SST/100
n_RMSE = numpy.sqrt(SST)

plt.plot(n_eval, n_pred, 'bo')
plt.plot(n_eval, n_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual n_p value')
plt.ylabel('Predicted n_p value')
plt.text(1.6,2.3,'RMSE=%s'%n_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()

plt.plot(a_pred, numpy.abs(diffn), 'bo')
plt.title('Radius vs n error')
plt.show()
