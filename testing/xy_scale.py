from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Cropping2D
from keras import backend as K
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import axes3d

print('Opening data...')

with open('../datasets/labdata_xy/eval_crop/' + 'data.txt', 'r') as file:
        evalset = json.load(file)


img_eval = []
rx_eval = []
ry_eval = []
z_eval = []
a_eval = evalset['a_p']
r_p = evalset['r_p']
files = evalset['filename']
for i in range(len(files)):
    stri = str(i+1)
    filestr = '../datasets/labdata_xy/eval_crop/img' + stri + '.png'
    im = Image.open(filestr)
    im_np = numpy.array(im)
    img_eval.append(im_np)
    im.close()
    rx_eval.append(r_p[i][0])
    ry_eval.append(r_p[i][1])
    z_eval.append(r_p[i][2])

img_eval = numpy.array(img_eval).astype('float32')
img_eval *= 1./255
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
xy_model = load_model('../models/predict_xy5.h5')
xy_model.summary()

#Localization
(x_pred,y_pred) = xy_model.predict(img_eval)

tol=100
xmax = img_cols/2-tol
xmin = -xmax
ymax = img_rows/2-tol
ymin = -ymax

x_pred = rescale_back(xmin, xmax, x_pred)
y_pred = rescale_back(ymin, ymax, y_pred)

diffx = []
diffy = []
for i in range(len(files)):
        diffy.append(numpy.abs(y_pred[i] - ry_eval[i]))
        diffx.append(numpy.abs(x_pred[i] - rx_eval[i]))

diffx = numpy.array(diffx)
diffy = numpy.array(diffy)

#plt.plot(diffx,diffy, 'bo')
#plt.show()

plotx = []
rx_plot = []
ploty = []
ry_plot = []
for i in range(len(files)):
        if diffx[i] < 300:
            plotx.append(x_pred[i])
            rx_plot.append(rx_eval[i])
        if diffy[i] < 200:
            ploty.append(y_pred[i])
            ry_plot.append(ry_eval[i])


plotx = numpy.array(plotx).astype('float32')
rx_plot =  numpy.array(rx_plot).astype('float32')
ploty =  numpy.array(ploty).astype('float32')
ry_plot =  numpy.array(ry_plot).astype('float32')
 
plotx = plotx.reshape(len(plotx))
ploty = ploty.reshape(len(ploty))

print(plotx.shape)

xpoly = numpy.polyfit(plotx, rx_plot, 1)
p_x = numpy.poly1d(xpoly)
polyfitx = p_x(x_pred)
print(p_x)

plt.plot(rx_eval, polyfitx, 'bo')
plt.plot(rx_eval, rx_eval, 'r')
#plt.plot(rx_eval, x_pred, 'bo')
plt.show()

ypoly = numpy.polyfit(ploty, ry_plot, 1)
p_y = numpy.poly1d(ypoly)
polyfity = p_y(y_pred)
print(p_y)

plt.plot(ry_eval, polyfity, 'bo')
plt.plot(ry_eval, ry_eval, 'r')
#plt.plot(ry_plot, ploty, 'bo')
plt.show()

