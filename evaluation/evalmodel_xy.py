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

file_head = '../datasets/xydata_biga/'

print('Opening data...')

with open(file_head + 'eval/' + 'data.txt', 'r') as file:
        evalset = json.load(file)


img_eval = []
z_eval = []
x_eval = []
y_eval = []
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
    x_eval.append(r_p[i][0])
    y_eval.append(r_p[i][1])
    z_eval.append(r_p[i][2])


img_eval = numpy.array(img_eval).astype('float32')
img_eval*= 1./255
x_eval = numpy.array(x_eval).astype('float32')
y_eval = numpy.array(y_eval).astype('float32')
z_eval = numpy.array(z_eval).astype('float32')
a_eval = numpy.array(a_eval).astype('float32')



img_rows = 480
img_cols = 640

if K.image_data_format() == 'channels_first':
    img_eval = img_eval.reshape(img_eval.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    img_eval = img_eval.reshape(img_eval.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



model = load_model('../models/predict_xy5.h5')

model.summary()

target = model.predict(img_eval)
x_pred = target[0]
y_pred = target[1]

def rescale_back(min, max, list):
    scalar = (max-min)/1.
    list = list*scalar +min
    return list

tol=100
xmax = img_cols/2-tol
xmin = -xmax
ymax = img_rows/2-tol
ymin = -ymax

x_pred = rescale_back(xmin, xmax, x_pred)
y_pred = rescale_back(ymin, ymax, y_pred)
#z_pred = rescale_back(50, 600, z_pred)
#a_pred  = rescale_back(0.2, 5, a_pred)
#n_pred = rescale_back(1.38, 2.5, n_pred) 

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}',
                                       r'\usepackage{sansmath}',
                                       r'\sansmath',
                                       r'\usepackage{siunitx}',
                                       r'\sisetup{detect-all}']

ticklabelsize = 20
axislabelsize = 30


diffx = numpy.zeros(100)
for i in range(0, 100):
    diffx[i] = x_eval[i] - x_pred[i]
sqerx = numpy.zeros(100)
for i in range(0, 100):
    sqerx[i] = diffx[i]**2
SST = numpy.sum(sqerx)
SST = SST/100
x_RMSE = numpy.sqrt(SST)


#xpoly = numpy.polyfit(x_eval, x_pred, 1)[:,0]
#p_x = numpy.poly1d(xpoly)
#polyfitx = p_x(x_eval)
#print(p_x)

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(x_eval, x_pred, 'bo')
ax.plot(x_eval, x_eval, 'r')
ax.set_xlabel(r'$x_p$ [pixel]', fontsize=axislabelsize)
ax.set_ylabel(r'$\overline{x_p}$ [pixel]', fontsize=axislabelsize)
plt.text(xmin+30,xmax-30,'$\Delta x_p$=%s px'%round(x_RMSE,1), bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=2)
#plt.show()

plt.savefig('../plots/franken_xacc.png', bbox_inches='tight')

'''
plt.plot(x_eval, polyfitx, 'g')
plt.plot(x_eval, x_pred, 'bo')
plt.plot(x_eval, x_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual x-value (pixels)')
plt.ylabel('Predicted x-value (pixels)')
plt.text(xmin+30,xmax-30,'RMSE=%s'%x_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()

x_scale = []
for root in x_pred:
    scaled_x = (p_x-root).roots
    x_scale.append(scaled_x[0])
x_scale = numpy.array(x_scale)


diffx = numpy.zeros(100)
for i in range(0, 100):
    diffx[i] = x_eval[i] - x_scale[i]
sqerx = numpy.zeros(100)
for i in range(0, 100):
    sqerx[i] = diffx[i]**2
SST = numpy.sum(sqerx)
SST = SST/100
x_RMSE = numpy.sqrt(SST)


plt.plot(x_eval, x_scale, 'go')
plt.plot(x_eval, x_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual x-value (pixels)')
plt.ylabel('Predicted x-value (pixels)')
plt.text(xmin+30,xmax-30,'RMSE=%s'%x_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()

'''
diffy = numpy.zeros(100)
for i in range(0, 100):
    diffy[i] = y_eval[i] - y_pred[i]
sqery = numpy.zeros(100)
for i in range(0, 100):
    sqery[i] = (diffy[i])**2
SST = numpy.sum(sqery)
SST = SST/100
y_RMSE = numpy.sqrt(SST)


fig, ax = plt.subplots(figsize=(7,7))
ax.plot(y_eval, y_pred, 'bo')
ax.plot(y_eval, y_eval, 'r')
ax.set_xlabel(r'$y_p$ [pixel]', fontsize=axislabelsize)
ax.set_ylabel(r'$\overline{y_p}$ [pixel]', fontsize=axislabelsize)
plt.text(ymin+30,ymax-30,'$\Delta y_p$=%s px'%round(y_RMSE,1), bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=2)
#plt.show()

plt.savefig('../plots/franken_yacc.png', bbox_inches='tight')


#ypoly = numpy.polyfit(y_eval, y_pred, 1)[:,0]
#p_y = numpy.poly1d(ypoly)
#polyfity = p_y(y_eval)
'''
#plt.plot(y_eval, polyfity, 'g')
plt.plot(y_eval, y_pred, 'bo')
plt.plot(y_eval, y_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual y-value (pixels)')
plt.ylabel('Predicted y-value (pixels)')
plt.text(ymin+30,ymax-30,'RMSE=%s'%y_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()


y_scale = []
for root in y_pred:
    scaled_y = (p_y-root).roots
    y_scale.append(scaled_y[0])
y_scale = numpy.array(y_scale)


diffy = numpy.zeros(100)
for i in range(0, 100):
    diffy[i] = y_eval[i] - y_scale[i]
sqery = numpy.zeros(100)
for i in range(0, 100):
    sqery[i] = diffy[i]**2
SST = numpy.sum(sqery)
SST = SST/100
y_RMSE = numpy.sqrt(SST)

plt.plot(y_eval, y_scale, 'go')
plt.plot(y_eval, y_eval, 'r')
plt.title('Model accuracy')
plt.xlabel('Actual y-value (pixels)')
plt.ylabel('Predicted y-value (pixels)')
plt.text(ymin+30,ymax-30,'RMSE=%s'%y_RMSE, bbox=dict(facecolor='white',alpha=0.5))
plt.show()


sqerr = numpy.zeros(100)
for i in range(0, 100):
    sqerr[i] = sqerx[i] + sqery[i]
SST = numpy.sum(sqerr)
SST = SST/100
r_RMSE = numpy.sqrt(SST)

plt.plot(z_eval, sqerr, 'bo')
plt.show()

plt.plot(n_eval, sqerr, 'bo')
plt.show()


plt.plot(a_eval, sqerr, 'bo')
plt.title('Model accuracy')
plt.xlabel('Radius of Particle (microns)')
plt.ylabel('Squared Radial Error of xy Prediction (pixels^2)')
plt.show()

for i in range(10):
    img_local = img_eval[i][:,:,0]
    img_local*= 255
    xlocal = x_scale[i]
    xx = [xlocal + img_cols/2]
    ylocal = y_scale[i]
    yy = [ylocal + img_rows/2]
    plt.imshow(img_local, cmap='gray')
    plt.plot(xx, yy, 'ro')
    plt.show()
'''
