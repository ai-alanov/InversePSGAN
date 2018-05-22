import os
import logging
from data_io import get_texture_iter
import utils


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return (zx - 1)*2**depth + 1


class Config(object):
    lr = 0.0002
    b1 = 0.5 # momentum term of adam
    l2_fac = 1e-8 # L2 weight regularization factor
    k = 1 # number of D updates vs G updates

    def __init__(self):
        self.nz_local = 30    
        self.nz_global = 60
        self.nz_periodic = 3
        self.nz_periodic_MLPnodes = 50
        self.nz = self.nz_local+self.nz_global+self.nz_periodic*2
        self.periodic_affine = False # if True planar waves sum x,y sinusoids,
                                     #  else axes aligned sinusoids x or y
        self.zx = 6
        self.zx_sample = 32 # size of the spatial dimension in Z
                            # for producing the samples
        self.zx_sample_quilt = self.zx_sample / 4

        self.nc = 3 # number of channels in input X (i.e. r,g,b)
        self.gen_ks = ([(5,5)] * 5)[::-1] # kernel sizes on each layer - should
                                          # be odd numbers for zero-padding stuff
        self.dis_ks = [(5,5)] * 5 # kernel sizes on each layer - should be
                                  # odd numbers for zero-padding stuff
        self.gen_ls = len(self.gen_ks) # num of layers in the generative network
        self.dis_ls = len(self.dis_ks) # num of layers in the discriminative network
        self.gen_fn = [self.nc]+[2**(n+6) for n in range(self.gen_ls-1)]
        # generative number of filters
        self.gen_fn = self.gen_fn[::-1]
        self.dis_fn = [2**(n+6) for n in range(self.dis_ls-1)]+[1] # discriminative
                                                                   # number of filters
        self.npx    = zx_to_npx(self.zx, self.gen_ls) # num of pixels width/height
                                                      # of images in X
        self.current_dir = os.path.abspath('.')
        self.sub_name    = "texture/"
        self.texture_dir = os.path.join(self.current_dir, self.sub_name)
        self.save_name = os.path.dirname(self.sub_name)
        self.save_name += "_filters%d_npx%d_%dgL_%ddL" \
                          % (self.dis_fn[0], self.npx, self.gen_ls, self.dis_ls)
        self.save_name += "_%dGlobal_%dPeriodic_%sAffine_%dLocal" \
                          % (self.nz_global, self.nz_periodic,
                             self.periodic_affine, self.nz_local)
        self.load_name   = None # if None, initializing network from scratch
           
    ## gives back the correct data iterator given class variables --
    ## this way we avoid the python restriction not to pickle iterator objects
    def data_iter(self):
        return get_texture_iter(self.texture_dir, npx=self.npx,
                                mirror=False, batch_size=self.batch_size)

    def print_info(self):
        logger = utils.create_logger('run_psgan.config')
        msg = "Learning and generating samples from zx {}, "
        msg += "which yields images of size npx {}"
        logger.info(msg.format(self.zx, zx_to_npx(self.zx, self.gen_ls)))
        msg = "Producing samples from zx_sample {}, "
        msg += "which yields images of size npx {}"
        logger.info(msg.format(self.zx_sample, zx_to_npx(self.zx_sample, self.gen_ls)))
        msg = "Saving samples and model data to file {}"
        logger.info(msg.format(self.save_name))

