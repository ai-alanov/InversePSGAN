import os
import numpy as np
from PIL import Image
from PIL.Image import FLIP_LEFT_RIGHT


def image_to_tensor(img):
    '''
    convert image to Theano/Lasagne 3-tensor format;
    changes channel dimension to be in the first position and rescales from [0,255] to [-1,1]
    '''
    tensor = np.array(img).transpose( (2,0,1) )
    tensor = (tensor / 255.)*2 - 1.
    return tensor


def tensor_to_image(tensor):
    '''
    convert 3-tensor to image;
    changes channel to be last and rescales from [-1, 1] to [0, 255]
    '''
    img = np.array(tensor).transpose( (1,2,0) )
    img = (img + 1.)/2 * 255.
    return np.uint8(img)
    

def get_texture_iter(texture_path, npx=128, batch_size=64, \
                     filter=None, mirror=True):
    '''
    @param folder       iterate of pictures from this folder
    @param npx          size of patches to extract
    @param n_batches    number of batches to yield - if None, it yields forever
    @param mirror       if True the images get augmented by left-right mirroring
    @return a batch of image patches fo size npx x npx, with values in [0,1]
    '''
    HW    = npx
    imgBig = None
    try:
        imgBig = Image.open(texture_path)
        if mirror:
            imgBig = imgBig.transpose(FLIP_LEFT_RIGHT)
    except:
        print "Image ", texture_path, " failed to load!"

    while True:
        data=np.zeros((batch_size,3,npx,npx))                   # NOTE: assumes 3 channels!
        for i in range(batch_size):
            if HW < imgBig.shape[1] and HW < imgBig.shape[2]:   # sample patches
                h = np.random.randint(imgBig.shape[1] - HW)
                w = np.random.randint(imgBig.shape[2] - HW)
                img = imgBig[:, h:h + HW, w:w + HW]
            else:                                               # whole input texture
                img = imgBig
            data[i] = img

        yield data


def save_tensor(tensor, filename):
    '''
    save a 3-tensor (channel, x, y) to image file
    '''
    img = tensor_to_image(tensor)
    img = Image.fromarray(img)
    img.save(filename)
