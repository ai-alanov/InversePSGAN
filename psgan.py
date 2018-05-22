import lasagne
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as Lrelu
from lasagne.nonlinearities import tanh, sigmoid
from lasagne.layers import batch_norm
from lasagne.layers import Conv2DLayer
from lasagne.layers import TransposedConv2DLayer

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import sys
import logging
from sklearn.externals import joblib

from config import Config
import utils


conv = lambda incoming, num_filters, filter_size, W, b, nonlinearity: \
    Conv2DLayer(incoming, num_filters, filter_size, stride=(2,2), pad='same',
                W=W, b=b, flip_filters=True, nonlinearity=nonlinearity)
tconv = lambda incoming, num_filters, filter_size, W, nonlinearity: \
    TransposedConv2DLayer(incoming, num_filters, filter_size, stride=(2,2),
                          crop='same', W=W, nonlinearity=nonlinearity)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

    
class PeriodicLayer(lasagne.layers.Layer):

    def __init__(self, incoming, config, wave_params):
        super(PeriodicLayer, self).__init__(incoming=incoming)
        self.config = config       
        self.wave_params = wave_params
        self.input_layer= incoming
        self.input_shape = incoming.output_shape
        self.get_output_kwargs = []
        self.params = {}
        for p in wave_params:
            self.params[p] = set('trainable')
        self.srng = RandomStreams(seed=1234)

    def _wave_calculation(self,Z):
        if self.config.nz_periodic ==0:
            return Z
        nPeriodic = self.config.nz_periodic

        if self.config.nz_global > 0:
            h = T.tensordot(Z[:, :self.config.nz_global],
                            self.wave_params[0], [1, 0])
            h = h.dimshuffle(0, 3, 1, 2)
            h += self.wave_params[1].dimshuffle('x', 0, 'x', 'x')

            band0 = T.tensordot(relu(h),self.wave_params[2], [1, 0])
            band0 = band0.dimshuffle(0, 3, 1, 2)
            band0 += self.wave_params[3].dimshuffle('x', 0, 'x', 'x')
        else:
            band0 = self.wave_params[0].dimshuffle('x', 0, 'x', 'x')
        
        if self.config.periodic_affine:
            band1 = Z[:, -nPeriodic*2::2] * band0[:, :nPeriodic]
            band1 += Z[:, -nPeriodic*2+1::2] * band0[:, nPeriodic:2*nPeriodic]
            band2 = Z[:, -nPeriodic*2::2] * band0[:, 2*nPeriodic:3*nPeriodic]
            band2 += Z[:, -nPeriodic*2+1::2] * band0[:, 3*nPeriodic:]
        else:
            band1 = Z[:, -nPeriodic*2::2] * band0[:, :nPeriodic]
            band2 = Z[:, -nPeriodic*2+1::2] * band0[:, 3*nPeriodic:]
        band = T.concatenate([band1 , band2], axis=1)
        offset =  2 * np.pi * self.srng.uniform((Z.shape[0], nPeriodic*2))
        offset = offset.dimshuffle(0, 1, 'x', 'x')
        band += offset
        return T.concatenate([Z[:, :-2*nPeriodic], T.sin(band)], axis=1)

    def get_output_for(self, input, **kwargs):
        return self._wave_calculation(input)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.config.nz_periodic,
                input_shape[2], input_shape[3])


periodic = lambda incoming, config, wave_params: \
    PeriodicLayer(incoming, config, wave_params)


class PSGAN(object):

    def __init__(self, name=None):
        self.w_init = lasagne.init.Normal(std=0.02)
        self.b_init = lasagne.init.Constant(val=0.0)
        self.g_init = lasagne.init.Normal(mean=1., std=0.02)

        if name is not None:
            print "loading parameters from file:", name

            vals =joblib.load(name)
            self.config = vals["config"]

            print "global dimensions of loaded config file", \
                self.config.nz_global

            self.dis_W = [sharedX(p) for p in vals["dis_W"]]
            self.dis_g = [sharedX(p) for p in vals["dis_g"]]
            self.dis_b = [sharedX(p) for p in vals["dis_b"]]

            self.gen_W = [sharedX(p) for p in vals["gen_W"]]
            self.gen_g = [sharedX(p) for p in vals["gen_g"]]
            self.gen_b = [sharedX(p) for p in vals["gen_b"]]

            self.wave_params = [sharedX(p) for p in vals["wave_params"]]

            self.config.gen_ks = []
            self.config.gen_fn = []
            l = len(vals["gen_W"])
            for i in range(l):
                if i == 0:
                    self.config.nz = vals["gen_W"][i].shape[0]
                else:
                    self.config.gen_fn +=[vals["gen_W"][i].shape[0]]
                self.config.gen_ks += [(vals["gen_W"][i].shape[2],
                                        vals["gen_W"][i].shape[3])]
            self.config.nc = vals["gen_W"][i].shape[1]
            self.config.gen_fn += [self.config.nc]

            self.config.dis_ks = []
            self.config.dis_fn = []
            l = len(vals["dis_W"])
            for i in range(l):
                self.config.dis_fn += [vals["dis_W"][i].shape[1]]
                self.config.dis_ks += [(vals["gen_W"][i].shape[2],
                                        vals["gen_W"][i].shape[3])]

            self._setup_gen_params(self.config.gen_ks, self.config.gen_fn)
            self._setup_dis_params(self.config.dis_ks, self.config.dis_fn)
        else:
            self.config = Config()

            self._setup_gen_params(self.config.gen_ks, self.config.gen_fn)
            self._setup_dis_params(self.config.dis_ks, self.config.dis_fn)
            self._sample_initials()

            self._setup_wave_params()

        self._build_sgan()


    def save(self,name):
        logger = logging.getLogger('run_psgan.psgan_save')
        logger.info("saving PSGAN parameters in file: {}".format(name))
        vals = {}
        vals["config"] = self.config
        vals["dis_W"] = [p.get_value() for p in self.dis_W]
        vals["dis_g"] = [p.get_value() for p in self.dis_g]
        vals["dis_b"] = [p.get_value() for p in self.dis_b]

        vals["gen_W"] = [p.get_value() for p in self.gen_W]
        vals["gen_g"] = [p.get_value() for p in self.gen_g]
        vals["gen_b"] = [p.get_value() for p in self.gen_b]

        vals["wave_params"] = [p.get_value() for p in self.wave_params]

        joblib.dump(vals, name, True)


    def _setup_wave_params(self):

        if self.config.nz_periodic:
            nPeriodic = self.config.nz_periodic
            nperiodK = self.config.nz_periodic_MLPnodes
            if self.config.nz_global > 0 and nperiodK > 0:
                lin1 = sharedX(self.g_init.sample((self.config.nz_global,nperiodK)))
                bias1 = sharedX(self.g_init.sample((nperiodK)))
                lin2 = sharedX(self.g_init.sample((nperiodK,nPeriodic*2*2)))
                bias2 = sharedX(self.g_init.sample((nPeriodic*2*2)))
                self.wave_params = [lin1, bias1, lin2, bias2]
            else:
                bias2 = sharedX(self.g_init.sample((nPeriodic*2*2)))
                self.wave_params = [bias2]
            a = np.zeros(nPeriodic*2*2)
            a[:nPeriodic] = 1
            a[nPeriodic:2*nPeriodic] = 0
            a[2*nPeriodic:3*nPeriodic] = 0
            a[3*nPeriodic:] = 1
            self.wave_params[-1].set_value(np.float32(a))
        else:
            self.wave_params = []

    def _setup_gen_params(self, gen_ks, gen_fn):
        if gen_ks == None:
            self.gen_ks = [(5,5)] * 5
        else:
            self.gen_ks = gen_ks


        self.gen_depth = len(self.gen_ks)

        if gen_fn!=None:
            assert len(gen_fn) == len(self.gen_ks), \
                'Layer number of filter numbers and sizes does not match.'
            self.gen_fn = gen_fn
        else:
            self.gen_fn = [64] * self.gen_depth

    def _setup_dis_params(self, dis_ks, dis_fn):
        if dis_ks == None:
            self.dis_ks = [(5,5)] * 5
        else:
            self.dis_ks = dis_ks

        self.dis_depth = len(dis_ks)

        if dis_fn != None:
            assert len(dis_fn) == len(self.dis_ks), \
                'Layer number of filter numbers and sizes does not match.'
            self.dis_fn = dis_fn
        else:
            self.dis_fn = [64] * self.dis_depth

    def _sample_initials(self):
        self.dis_W = []
        self.dis_b = []
        self.dis_g = []


        self.dis_W.append(sharedX(self.w_init.sample(
            (self.dis_fn[0], self.config.nc,
             self.dis_ks[0][0], self.dis_ks[0][1]))))
        for l in range(self.dis_depth-1):
            self.dis_W.append(sharedX(self.w_init.sample(
                (self.dis_fn[l+1], self.dis_fn[l],
                 self.dis_ks[l+1][0], self.dis_ks[l+1][1]))))
            self.dis_b.append(sharedX(self.b_init.sample((self.dis_fn[l+1]))))
            self.dis_g.append(sharedX(self.g_init.sample((self.dis_fn[l+1]))))

        self.gen_b = []
        self.gen_g = []
        for l in range(self.gen_depth - 1):
            self.gen_b += [sharedX(self.b_init.sample((self.gen_fn[l])))]
            self.gen_g += [sharedX(self.g_init.sample((self.gen_fn[l])))]

        self.gen_W = []

        last = self.config.nz
        for l in range(self.gen_depth - 1):
            self.gen_W += [sharedX(self.w_init.sample(
                (last,self.gen_fn[l], self.gen_ks[l][0], self.gen_ks[l][1])))]
            print last,self.gen_fn[l], self.gen_ks[l][0], self.gen_ks[l][1]
            last = self.gen_fn[l]

        self.gen_W += [sharedX(self.w_init.sample(
            (last,self.gen_fn[-1], self.gen_ks[-1][0],self.gen_ks[-1][1])))]
        print last,self.gen_fn[-1], self.gen_ks[-1][0],self.gen_ks[-1][1]

    def _spatial_generator(self, inlayer):
        layers = [inlayer]
        layers.append(periodic(inlayer, self.config, self.wave_params))
        for l in range(self.gen_depth - 1):
            layers.append(batch_norm(
                tconv(layers[-1], self.gen_fn[l], self.gen_ks[l],
                      self.gen_W[l], nonlinearity=relu),
                gamma=self.gen_g[l], beta=self.gen_b[l]))
        output  = tconv(layers[-1], self.gen_fn[-1], self.gen_ks[-1],
                        self.gen_W[-1], nonlinearity=tanh)

        return output

    def _spatial_discriminator(self, inlayer):
        layers  = [inlayer]
        layers.append(conv(layers[-1], self.dis_fn[0], self.dis_ks[0],
                           self.dis_W[0], None, nonlinearity=Lrelu(0.2)))
        for l in range(1,self.dis_depth - 1):
            layers.append(batch_norm(
                conv(layers[-1], self.dis_fn[l], self.dis_ks[l],
                     self.dis_W[l], None, nonlinearity=Lrelu(0.2)),
                gamma=self.dis_g[l-1], beta=self.dis_b[l-1]))
        output = conv(layers[-1], self.dis_fn[-1], self.dis_ks[-1],
                      self.dis_W[-1], None, nonlinearity=sigmoid)

        return output

    def _build_sgan(self):
        Z = lasagne.layers.InputLayer((None, self.config.nz, None, None))
        X = lasagne.layers.InputLayer((None, self.config.nc,
                                       self.config.npx, self.config.npx))

        gen_X = self._spatial_generator(Z)
        d_real = self._spatial_discriminator(X)
        d_fake = self._spatial_discriminator(gen_X)

        prediction_gen = lasagne.layers.get_output(gen_X)
        prediction_real = lasagne.layers.get_output(d_real)
        prediction_fake = lasagne.layers.get_output(d_fake)

        params_g = lasagne.layers.get_all_params(gen_X, trainable=True)
        params_d = lasagne.layers.get_all_params(d_real, trainable=True)

        l2_gen = lasagne.regularization.regularize_network_params(
            gen_X, lasagne.regularization.l2)
        l2_dis = lasagne.regularization.regularize_network_params(
            d_real, lasagne.regularization.l2)

        obj_d= -T.mean(T.log(1-prediction_fake)) \
               - T.mean( T.log(prediction_real)) + self.config.l2_fac * l2_dis
        obj_g= -T.mean(T.log(prediction_fake)) + self.config.l2_fac * l2_gen

        updates_d = lasagne.updates.adam(obj_d, params_d,
                                         self.config.lr, self.config.b1)
        updates_g = lasagne.updates.adam(obj_g, params_g,
                                         self.config.lr, self.config.b1)

        logger = utils.create_logger('run_psgan.psgan_build', stream=sys.stdout)
        logger.info("Compiling the network...")
        self.train_d = theano.function(
            [X.input_var, Z.input_var], obj_d,
            updates=updates_d, allow_input_downcast=True)
        logger.info("Discriminator done.")
        self.train_g = theano.function(
            [Z.input_var], obj_g, updates=updates_g, allow_input_downcast=True)
        logger.info("Generator done.")
        self.generate = theano.function(
            [Z.input_var], prediction_gen, allow_input_downcast=True)
        logger.info("generate function done.")
