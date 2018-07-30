from __future__ import print_function
import keras, json, numpy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf


# input image dimensions
img_rows, img_cols = 201,201

#unpack data
print('Unpacking data...')
with open('../datasets/data.txt', 'r') as file:
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

def baseline_model():
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
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    Adamlr = keras.optimizers.Adam(lr=0.0007)
    model.compile(loss='mean_squared_error', optimizer=Adamlr)
    return model



'''
#Cross-validation
seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=1)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''


#Fitting Parameters
batch_size = 32
epochs = 200
seed=7

#Callbacks
callbacks = []

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
earlystopCB = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=2000, patience=10, verbose=0, mode='auto', baseline=None)

'''
#custom callback for image visualization in tensorboard
def make_image(tensor):
        """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
        from PIL import Image
        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,                                                                            width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

        def on_epoch_end(self, epoch, logs={}):
            XX = self.input
            YY = self.layers[0].output
            intermediate_model = Model(XX,YY)
            Xaug = x_train[:9]
            # Load image
            img = intermediate_model.predict(Xaug)[0]
            # Do something to the image
            #img = (255 * skimage.util.random_noise(img)).astype('uint8')

            image = make_image(img)
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
            writer = tf.summary.FileWriter('./logs')
            writer.add_summary(summary, epoch)
            writer.close()

            return

tbi_callback = TensorBoardImage('Image Example')
'''

callbacks.append(tbCallBack)
#callbacks.append(earlystopCB)


estimator = baseline_model()
'''
kf = KFold(n_splits=5, random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv=kf)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''
estimator.summary()

estimator.fit(x_train, y_train,
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = (x_test, y_test),
              callbacks=callbacks)

bias = estimator.layers[-1].get_weights()[1]
scores = estimator.evaluate(x=x_test, y=y_test, verbose=1)
print("Final bias", bias)
print("MSE", scores)
estimator.save('../models/predict_z.h5')

#estimator.fit(x_train, y_train)
#y_pred = estimator.predict(x_test)
#plt.plot(y_test, y_pred, 'ro')
#plt.plot(y_test, y_test)
#plt.show()


#estimator_json = estimator.model.to_json()
#print(estimator_json)
#with open('json_model.txt', 'w') as outfile:
#        json.dump(estimator_json, outfile)
