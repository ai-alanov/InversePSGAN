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


def get_images(img_files):
    imgs = []
    for file in img_files:
        try:
            img = Image.open(file)
            imgs += [image_to_tensor(img)]
        except:
            print "Image ", file, " failed to load!"
    return imgs
    

def get_texture_iter(texture_path, npx=128, batch_size=64, mirror=False):
    HW = npx
    imTex = []
    try:
        files = os.listdir(texture_path)
        files = [texture_path + file for file in files]
    except:
        files = [texture_path]
    for file in files:
        try:
            img = Image.open(file)
            imTex += [image_to_tensor(img)]
            if mirror:
                img = img.transpose(FLIP_LEFT_RIGHT)
                imTex += [image_to_tensor(img)]
        except:
            print "Image ", file, " failed to load!"

    while True:
        data = np.zeros((batch_size, 3, npx, npx))  # NOTE: assumes 3 channels!
        for i in range(batch_size):
            ir = np.random.randint(len(imTex))
            imgBig = imTex[ir]
            if HW < imgBig.shape[1] and HW < imgBig.shape[2]:  # sample patches
                h = np.random.randint(imgBig.shape[1] - HW)
                w = np.random.randint(imgBig.shape[2] - HW)
                img = imgBig[:, h:h + HW, w:w + HW]
            else:
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
