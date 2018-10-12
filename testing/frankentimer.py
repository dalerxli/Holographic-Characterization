
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Cropping2D
from keras import backend as K
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib import patches

import timeit

mysetup = '''

import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Cropping2D
from keras import backend as K
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib import patches

file_head = '../datasets/xydata_biga/'

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
    #    a_eval.append(evalset['a_p'][i])


img_eval = numpy.array(img_eval).astype('float32')
img_eval*= 1./255
z_eval = numpy.array(z_eval).astype('float32')
a_eval = numpy.array(a_eval).astype('float32')
n_eval = numpy.array(n_eval).astype('float32')


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

img_eval = format_image(img_eval)

def rescale(min, max, target, list):
    scalar = 1./(max-min)
    list = (list-min)*scalar
    return list

def rescale_back(min, max, list):
    scalar = (max-min)/1.
    list = list*scalar + min
    return list


#Load Previous Models
stamp_model = load_model('../models/predict_stamp_overtrain2.h5')
xy_model = load_model('../models/predict_xy5.h5')'''
mycode = '''
#Localization
(x_pred,y_pred) = xy_model.predict(img_eval)

tol=100
xmax = img_cols/2-tol
xmin = -xmax
ymax = img_rows/2-tol
ymin = -ymax

x_pred = rescale_back(xmin, xmax, x_pred)
y_pred = rescale_back(ymin, ymax, y_pred)

#Cropping
crop_img = []
for i in range(100):
    xreal = rx_eval[i]
    yreal = ry_eval[i]
    (xc, yc) = x_pred[i][0], y_pred[i][0]

    xc = xc + img_cols/2
    yc = yc + img_rows/2
   
    img_local = img_eval[i][:,:,0]
    
    xc = int(round(xc))
    yc = int(round(yc))
    xtop = xc+ tol+1
    xbot = xc - tol
    ytop = yc +tol+1
    ybot = yc-tol
    wid = xtop-xbot
    hig = ytop - ybot
    
    if xbot<0:
        xbot = 0
        xtop = 201
    if ybot<0:
        ybot = 0
        ytop = 201
    if xtop>img_cols:
        xtop = img_cols
        xbot = img_cols - 201
    if ytop>img_rows:
        ytop = img_rows
        ybot = img_rows - 201

    cropped = img_local[ybot:ytop, xbot:xtop]
    crop_img.append(cropped)

img_rows, img_cols = 201, 201
crop_img = numpy.array(crop_img)
crop_img = format_image(crop_img)

#Characterization
char = stamp_model.predict(crop_img)
z_pred = rescale_back(50, 600, char[0])
a_pred = rescale_back(0.2, 5., char[1])
n_pred = rescale_back(1.38, 2.5, char[2])
'''


ntimes = 10
time100 = timeit.timeit(setup = mysetup, stmt = mycode, number=ntimes)/ntimes
time1 = time100/100
print(time100, 'all')
print(time1, '1')

