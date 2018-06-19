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


def get_images(img_files, mirror=False):
    imgs = []
    for file in img_files:
        try:
            img = Image.open(file)
            imgs += [image_to_tensor(img)]
            if mirror:
                img = img.transpose(FLIP_LEFT_RIGHT)
                imgs += [image_to_tensor(img)]
        except Exception:
            print "Image ", file, " failed to load!"
    return imgs


def apply_random_rotate(img):
    n_rotations = np.random.randint(4)
    img = np.rot90(img, k=n_rotations, axes=(1, 2))
    return img


def apply_random_flip(img):
    img = img.transpose((1, 2, 0))
    if np.random.randint(2):
        img = np.fliplr(img)
    if np.random.randint(2):
        img = np.flipud(img)
    img = img.transpose((2, 0, 1))
    return img


def get_random_patch(imgBig, HW):
    if HW < imgBig.shape[1] and HW < imgBig.shape[2]:
        h = np.random.randint(imgBig.shape[1] - HW)
        w = np.random.randint(imgBig.shape[2] - HW)
        img = imgBig[:, h:h + HW, w:w + HW]
    else:
        img = imgBig
    img = apply_random_rotate(img)
    img = apply_random_flip(img)
    return img


def get_texture_iter(texture_path, npx=128, batch_size=64,
                     mirror=False, inverse=0):
    HW = npx
    try:
        files = os.listdir(texture_path)
        files = [texture_path + file for file in files]
    except Exception:
        files = [texture_path]
    imTex = get_images(files, mirror=mirror)

    while True:
        data = np.zeros((batch_size, 3, npx, npx))
        if inverse >= 2:
            data2 = np.zeros((batch_size, 3, npx, npx))
        for i in range(batch_size):
            ir = np.random.randint(len(imTex))
            imgBig = imTex[ir]
            data[i] = get_random_patch(imgBig, HW)
            if inverse >= 2:
                data2[i] = get_random_patch(imgBig, HW)
        if inverse >= 2:
            yield data, data2
        else:
            yield data


def save_tensor(tensor, filename):
    '''
    save a 3-tensor (channel, x, y) to image file
    '''
    img = tensor_to_image(tensor)
    img = Image.fromarray(img)
    img.save(filename)
