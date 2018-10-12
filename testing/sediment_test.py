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

numfiles = 1096

img_eval = []
for i in range(numfiles):
    stri = str(i+1)
    stri = stri.zfill(4)
    filestr = '../datasets/sediment_mov/norm_img/img' + stri + '.png'
    im = Image.open(filestr)
    im_np = numpy.array(im)
    img_eval.append(im_np)
    im.close()

img_eval = numpy.array(img_eval).astype('float32')
img_eval*= 1./255

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
stamp_model = load_model('../models/predict_lab_stamp.h5')
xy_model = load_model('../models/predict_xy5.h5')
stamp_model.summary()
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


#Cropping
crop_img = []
for i in range(len(y_pred)):
    
    #xreal = rx_eval[i]
    #yreal = ry_eval[i]
    (xc, yc) = x_pred[i][0], y_pred[i][0]

    '''
    if numpy.abs(xc - xreal) > 30:
        continue
    if numpy.abs(yc - yreal) > 30:
        continue
    '''
    xc = xc + img_cols/2
    yc = yc + img_rows/2
   
    img_local = img_eval[i][:,:,0]
    img_local*= 255
    
    xc = int(round(xc))
    yc = int(round(yc))
    xtop = xc+ tol+1
    xbot = xc - tol
    ytop = yc +tol+1
    ybot = yc-tol
    wid = xtop-xbot
    hig = ytop - ybot
    
    if i in range(5):
        fig,ax = plt.subplots(1)
        ax.imshow(img_local, cmap='gray')
        ax.plot([xc], [yc], 'ro')
    
        rect = patches.Rectangle((xbot, ybot), wid, hig, ec='red', fill=False)
        ax.add_patch(rect)
        plt.show()
    
    
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

    if i== -1:
        #With PIL instead
        pilim = img_local
        data_u8 = pilim.astype('uint8')
        pilim = Image.fromarray(data_u8 , 'L')
        draw = ImageDraw.Draw(pilim)
        #draw.point((xc, yc), 'red')
        #pilim.show()
        
        cropped = pilim.crop((xbot, ybot, xtop, ytop))
        cropped.show()
    
    
    cropped = img_local[ybot:ytop, xbot:xtop]
    '''
    if i ==4:
        plt.imshow(cropped, cmap='gray')
        plt.show()
    '''
    cropped *= 1./255
    crop_img.append(cropped)
    
img_rows, img_cols = 201, 201
crop_img = numpy.array(crop_img)
crop_img = format_image(crop_img)

#Characterization
char = stamp_model.predict(crop_img)
z_pred = rescale_back(50, 600, char[0])
a_pred = rescale_back(0.2, 5., char[1])
a_mean = numpy.mean(a_pred)
n_pred = rescale_back(1.38, 2.5, char[2])
n_mean = numpy.mean(n_pred)

#Rescale (x,y) predictions
x_pred = x_pred*2 - 14
y_pred = y_pred*2 - 31

data = {'x': x_pred.tolist(), 'y': y_pred.tolist(), 'z': z_pred.tolist(), 'a': a_pred.tolist(), 'n': n_pred.tolist()}
with open('sediment_predictions.json', 'w') as outfile:
    json.dump(data, outfile)

time = numpy.linspace(0, 100,num=numfiles)
plt.scatter(time,a_pred)
plt.plot(time, a_mean*numpy.ones(numfiles), 'r')
plt.title('Radius predictions')
plt.show()
plt.scatter(time,n_pred)
plt.plot(time, n_mean*numpy.ones(numfiles), 'r')
plt.title('Refractive index predictions')
plt.show()

plt.scatter(time, z_pred)
plt.title('Height predictions')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x_pred, y_pred, z_pred, c='b')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D position over time')
plt.show()

