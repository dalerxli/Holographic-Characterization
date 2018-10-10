import numpy as np
import sys
#or wherever you have theory stored
sys.path.append('/home/lea336/lorenzmie/theory')
from spheredhm import spheredhm
import json, base64
#import matplotlib.pyplot as plt
from keras import backend as K
from PIL import Image

#values that won't change
xpix = 201
ypix = 201
dim = [xpix, ypix]
mpp = 0.135
n_m = 1.339
lamb = 0.447

#Add gaussian noise to data                                                
def noisy(img):
    row,col,ch= ypix, xpix, 1
    mean=0
    var=0.05
    sigma=var**0.5
    gauss=np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy

def format_image(img):
    if K.image_data_format() == 'channels_first':
        img = img.reshape(1, ypix, xpix)
    else:
        img = img.reshape(ypix, xpix, 1)
    return img

tol = 100

def createdata(n_samp, folder_name): 
    Xset = []
    Yset = []
    for i in range(0, n_samp):
        #z = 10.0/mpp
        z = np.random.uniform(low= 50, high = 600)
        #x, y = 0,0
        x = np.random.normal(0,5)
        y = np.random.normal(0,5)
#        x = np.random.uniform(low=(-xpix/2 + tol), high = (xpix/2-tol))
#        y = np.random.uniform(low=(-ypix/2 + tol), high = (ypix/2-tol))
        rp = [x,y,z]                 
        a_p = np.random.uniform(low= 0.2, high = 5.)
        n_p = np.random.uniform(low=1.38, high=2.5)
        image = spheredhm(rp, a_p, n_p, n_m , dim, lamb = lamb, mpp = mpp)
        image = format_image(image)
        image = image.astype('float32')
        image = noisy(image)
        image = image*100.
        pilim = image[:,:,0]
        np.clip(pilim, 0, 255, out=pilim)
        data_u8 = pilim.astype('uint8')
        pilim = Image.fromarray(data_u8 , 'L')
        pilim.show()
        filename = folder_name +'/img' + str(i+1)+'.png'
        pilim.save(filename)
        target = [rp,a_p,n_p]
        Xset.append(filename)
        Yset.append(target)
        print('Completed', i+1, 'images', end='\r')
    return [Xset, Yset]

[img, target] = createdata(3, 'none')

print(target)
'''
if __name__ == '__main__':
    print('Training set')
    train_num = 10000
    test_num = 1000
    eval_num = 100
    training_set = createdata(train_num, 'train')
    train_files = training_set[0]
    train_params = training_set[1]
    train_pos = [item[0] for item in train_params]
    train_rad = [item[1] for item in train_params]
    train_n = [item[2] for item in train_params]
    data = {'filename': train_files, 'r_p':train_pos, 'a_p':train_rad, 'n_p': train_n}
    with open('train/data.txt', 'w') as outfile:
        json.dump(data, outfile)
    print('Saved json')
    print('Test set')
    test_set = createdata(test_num, 'test')
    test_files = test_set[0]
    test_params = test_set[1]
    test_pos = [item[0] for item in test_params]
    test_rad = [item[1] for item in test_params]
    test_n = [item[2] for item in test_params]
    data = {'filename': test_files, 'r_p':test_pos, 'a_p':test_rad, 'n_p': test_n}
    with open('test/data.txt', 'w') as outfile:
        json.dump(data, outfile)
    print('Saved json')
    print('Eval set')
    eval_set = createdata(eval_num, 'eval')
    
    eval_files = eval_set[0]
    eval_params = eval_set[1]
    eval_pos = [item[0] for item in eval_params]
    eval_rad = [item[1] for item in eval_params]
    eval_n = [item[2] for item in eval_params]
    
    data = {'filename': eval_files, 'r_p':eval_pos, 'a_p':eval_rad, 'n_p': eval_n}
    with open('eval/data.txt', 'w') as outfile:
        json.dump(data, outfile)
    print('Saved json')
    print('Finished')
                                                                                                                                                                                
'''
