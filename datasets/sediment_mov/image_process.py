import numpy
from keras import backend as K
from PIL import Image

numfiles = 1096


#Open
img_eval =[]
min_cand = []
file_head = './images/img'
for i in range(numfiles):
    stri = str(i+1)
    stri = stri.zfill(4)
    im = Image.open(file_head + stri+ '.png')
    im_np = numpy.array(im)
    #Convert to Grayscale
    im_np = im_np[:,:,0]
    img_eval.append(im_np)
    min_cand.append(im_np.min())
    im.close()
    
img_eval = numpy.array(img_eval).astype('float32')
min_cand = numpy.array(min_cand).astype('float32')
dark = min_cand.min()
print(img_eval.shape)

#Normalize
bg_img = numpy.zeros(img_eval[0].shape)
for i in range(len(bg_img)):
    for j in range(len(bg_img[0])):
        pix_i = []
        for file in img_eval[0::10]:
            pix_i.append(file[i][j])
        pixel = numpy.median(pix_i)
        bg_img[i][j] = pixel

#bgim = Image.fromarray(numpy.uint8(bg_img))
#bgim.show()

img_save = []
for i in range(numfiles):
    testimg = img_eval[i][:]
    numer =testimg - dark
    denom = numpy.clip((bg_img-dark),1,255)
    testimg = numpy.divide(numer, denom)*100.
    testimg = numpy.clip(testimg, 0, 255)
    #Lower resolution
    testimg = testimg[:960:2,::2]
    img_save.append(testimg)

img_save =  numpy.array(img_save).astype('float32')
print(img_save.shape)

for i in range(numfiles):
    pilim = Image.fromarray(numpy.uint8(img_save[i]))
    stri = str(i+1)
    stri = stri.zfill(4)
    filestr = './norm_img/img' + stri + '.png'
    pilim.save(filestr)

    
pilim.show()



