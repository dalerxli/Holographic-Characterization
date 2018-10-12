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
stamp_model = load_model('../models/predict_stamp_overtrain3.h5')
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
z_test = []
a_test = []
n_test = []
for i in range(len(y_pred)):
    xreal = rx_eval[i]
    yreal = ry_eval[i]
    (xc, yc) = x_pred[i][0], y_pred[i][0]
    
    if numpy.abs(xc - xreal) > 30:
        continue
    if numpy.abs(yc - yreal) > 30:
        continue
    
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
    '''
    if i==4:
        fig,ax = plt.subplots(1)
        ax.imshow(img_local, cmap='gray')
        ax.plot([xc], [yc], 'ro')
    
        rect = patches.Rectangle((xbot, ybot), wid, hig, ec='red', fill=False)
        ax.add_patch(rect)
        plt.show()
    '''
    
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

    '''
    With PIL instead
    pilim = img_eval[i][:,:,0]*255
    data_u8 = pilim.astype('uint8')
    pilim = Image.fromarray(data_u8 , 'L')
    draw = ImageDraw.Draw(pilim)
    draw.point((xc, yc), 'red')
    #pilim.show()

    cropped = pilim.crop((xbot, ybot, xtop, ytop))
    cropped.show()
    '''
    
    cropped = img_local[ybot:ytop, xbot:xtop]
    '''
    if i ==4:
        plt.imshow(cropped, cmap='gray')
        plt.show()
    '''
    cropped *= 1./255
    crop_img.append(cropped)
    z_test.append(z_eval[i])
    a_test.append(a_eval[i])
    n_test.append(n_eval[i])

img_rows, img_cols = 201, 201
crop_img = numpy.array(crop_img)
crop_img = format_image(crop_img)
z_test = numpy.array(z_test)
a_test = numpy.array(a_test)
n_test = numpy.array(n_test)

#Characterization
char = stamp_model.predict(crop_img)
z_pred = rescale_back(50, 600, char[0])
a_pred = rescale_back(0.2, 5., char[1])
n_pred = rescale_back(1.38, 2.5, char[2])

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}',
                                       r'\usepackage{sansmath}',
                                       r'\sansmath',
                                       r'\usepackage{siunitx}',
                                       r'\sisetup{detect-all}']

ticklabelsize = 20
axislabelsize = 30

#Evaluation
diffz = numpy.zeros(100)
for i in range(0, 100):
    diffz[i] = z_test[i] - z_pred[i]
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = (diffz[i])**2
SST = numpy.sum(sqer)
SST = SST/100
z_RMSE = numpy.sqrt(SST)
#print('z RMSE', RMSE)

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(z_test, z_pred, 'bo')
ax.plot(z_eval, z_eval, 'r')
ax.set_xlabel(r'$z_p$ [pixel]', fontsize=axislabelsize)
ax.set_ylabel(r'$\overline{z_p}$ [pixel]', fontsize=axislabelsize)
plt.text(80,500,'$\Delta z_p$=%s px'%round(z_RMSE,1), bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=2)
#plt.show()

plt.savefig('../plots/franken_zacc.png', bbox_inches='tight')


diffa = numpy.zeros(100)
for i in range(0, 100):
    diffa[i] = a_test[i] - a_pred[i]
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = diffa[i]**2
SST = numpy.sum(sqer)
SST = SST/100
a_RMSE = numpy.sqrt(SST)

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(a_test, a_pred, 'bo')
ax.plot(a_eval, a_eval, 'r')
ax.set_xlabel(r'$a_p$ [\si{\um}]', fontsize=axislabelsize)
ax.set_ylabel(r'$\overline{a_p}$ [\si{\um}]', fontsize=axislabelsize)
plt.text(1.2,4.5,r'$\Delta a_p$=%s [\si{\um}]'%round(a_RMSE,2), bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=2)
#plt.show()

plt.savefig('../plots/franken_aacc.png', bbox_inches='tight')


diffn = numpy.zeros(100)
for i in range(0, 100):
    diffn[i] = n_test[i] - n_pred[i]
sqer = numpy.zeros(100)
for i in range(0, 100):
    sqer[i] = (diffn[i])**2
SST = numpy.sum(sqer)
SST = SST/100
n_RMSE = numpy.sqrt(SST)
#print('z RMSE', RMSE)

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(n_test, n_pred, 'bo')
ax.plot(n_eval, n_eval, 'r')
ax.set_xlabel(r'$n_p$', fontsize=axislabelsize)
ax.set_ylabel(r'$\overline{n_p}$', fontsize=axislabelsize)
plt.text(1.45,2.3,r'$\Delta n_p$=%s'%round(n_RMSE,2), bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=2)
#plt.show()

plt.savefig('../plots/franken_nacc.png', bbox_inches='tight')


print("It's alive!!")
