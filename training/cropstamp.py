from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Cropping2D
from keras import backend as K
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

file_head = '../datasets/xydata_biga/'

print('Opening data...')


with open(file_head + 'train/' + 'data.txt', 'r') as file:
    trainset = json.load(file)


img_train = []
rx_train = []
ry_train = []
z_train = []
r_p = trainset['r_p']
files = trainset['filename']
a_train = trainset['a_p']
n_train = trainset['n_p']
for i in range(len(files)):
    filename = file_head  + files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_train.append(im_np)
    im.close()
    rx_train.append(r_p[i][0])
    ry_train.append(r_p[i][1])
    z_train.append(r_p[i][2])
    

img_train = numpy.array(img_train).astype('float32')
img_train*= 1./255
z_train = numpy.array(z_train).astype('float32')
a_train = numpy.array(a_train).astype('float32')
n_train = numpy.array(n_train).astype('float32')



file_head = '../datasets/xydata_biga/'

print('Opening data...')


with open(file_head + 'test/' + 'data.txt', 'r') as file:
    testset = json.load(file)


img_test = []
rx_test = []
ry_test = []
z_test = []
r_p = testset['r_p']
files = testset['filename']
a_test = testset['a_p']
n_test = testset['n_p']
for i in range(len(files)):
    filename = file_head  + files[i]
    im = Image.open(filename)
    im_np = numpy.array(im)
    img_test.append(im_np)
    im.close()
    rx_test.append(r_p[i][0])
    ry_test.append(r_p[i][1])
    z_test.append(r_p[i][2])


img_test = numpy.array(img_test).astype('float32')
img_test*= 1./255
z_test = numpy.array(z_test).astype('float32')
a_test = numpy.array(a_test).astype('float32')
n_test = numpy.array(n_test).astype('float32')


img_rows = 480
img_cols = 640

def format_image(img):
    if K.image_data_format() == 'channels_first':
        img = img.reshape(img.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        img = img.reshape(img.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return(img)

img_train = format_image(img_train)
img_test = format_image(img_test)

def rescale(min, max, list):
    scalar = 1./(max-min)
    list = (list-min)*scalar
    return list

def rescale_back(min, max, list):
    scalar = (max-min)/1.
    list = list*scalar + min
    return list

z_train = rescale(50, 600, z_train)
z_test = rescale(50, 600, z_test)
a_train = rescale(0.2, 5., a_train)
a_test = rescale(0.2, 5., a_test)
n_train = rescale(1.38, 2.5, n_train)
n_test = rescale(1.38, 2.5, n_test)

#Load Previous Models
stamp_model = load_model('../models/predict_stamp_overtrain2.h5')
xy_model = load_model('../models/predict_xy4.h5')
stamp_model.summary()
xy_model.summary()

tol=100
xmax = img_cols/2-tol
xmin = -xmax
ymax = img_rows/2-tol
ymin = -ymax


(x_train_pred,y_train_pred) = xy_model.predict(img_train)
x_train_pred = rescale_back(xmin, xmax, x_train_pred)
y_train_pred = rescale_back(ymin, ymax, y_train_pred)

cropped_train = []
for i in range(len(img_train)):
    (xc, yc) = x_train_pred[i][0], y_train_pred[i][0]
    xc = xc + img_cols/2
    yc = yc + img_rows/2
    xtop = xc+ tol+1
    xbot = xc - tol
    ytop = yc +tol+1
    ybot = yc-tol
    
    pilim = img_train[i][:,:,0]*255
    data_u8 = pilim.astype('uint8')
    pilim = Image.fromarray(data_u8 , 'L')
    draw = ImageDraw.Draw(pilim)
    draw.point((xc, yc), 'red')
    #pilim.show()

    cropped = pilim.crop((xbot, ybot, xtop, ytop))
    #cropped.show()

    img_rows, img_cols = 201, 201
    
    cropped = numpy.array(cropped)
    cropped = numpy.array([cropped]).astype('float32')
    cropped *= 1./255
    cropped = format_image(cropped)
    cropped_train.append(cropped[0])

cropped_train = numpy.array(cropped_train)
img_rows, img_cols = 480,640

(x_test_pred,y_test_pred) = xy_model.predict(img_test)
x_test_pred = rescale_back(xmin, xmax, x_test_pred)
y_test_pred = rescale_back(ymin, ymax, y_test_pred)

cropped_test = []
for i in range(len(img_test)):
    (xc, yc) = x_test_pred[i][0], y_test_pred[i][0]
    xc = xc + img_cols/2
    yc = yc + img_rows/2
    xtop = xc+ tol+1
    xbot = xc - tol
    ytop = yc +tol+1
    ybot = yc-tol
    
    pilim = img_test[i][:,:,0]*255
    data_u8 = pilim.astype('uint8')
    pilim = Image.fromarray(data_u8 , 'L')
    draw = ImageDraw.Draw(pilim)
    draw.point((xc, yc), 'red')
    #pilim.show()

    cropped = pilim.crop((xbot, ybot, xtop, ytop))
    #cropped.show()

    img_rows, img_cols = 201, 201
    
    cropped = numpy.array(cropped)
    cropped = numpy.array([cropped]).astype('float32')
    cropped *= 1./255
    cropped = format_image(cropped)
    cropped_test.append(cropped[0])

cropped_test = numpy.array(cropped_test)
    
epochs = 100
batch_size = 16
stamp_model.fit({'image': cropped_train},
                {'z': z_train, 'a': a_train, 'n': n_train},
                epochs = epochs,
                batch_size = batch_size,
                verbose = 1,
                validation_data = ({'image': cropped_test},
                                   {'z': z_test, 'a': a_test, 'n': n_test}))

stamp_model.save('../models/cropped_stamp.h5')


