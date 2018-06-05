import lasagne
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as Lrelu
from lasagne.nonlinearities import tanh, sigmoid
from lasagne.layers import batch_norm
from lasagne.layers import Conv2DLayer
from lasagne.layers import TransposedConv2DLayer
from lasagne.layers import get_output, get_all_params
from lasagne.regularization import regularize_network_params

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gradient import disconnected_grad as zero_grad

import numpy as np
import sys
import logging
from sklearn.externals import joblib

from config import Config
import utils


conv = lambda incoming, num_filters, filter_size, W, b, nonlinearity, stride=2: \
    Conv2DLayer(incoming, num_filters, filter_size, stride=(stride,stride),
                pad='same', W=W, b=b, flip_filters=True, nonlinearity=nonlinearity)
tconv = lambda incoming, num_filters, filter_size, W, \
               nonlinearity, stride=2, crop='same': \
    TransposedConv2DLayer(incoming, num_filters, filter_size,
                          stride=(stride,stride), crop=crop, W=W,
                          nonlinearity=nonlinearity)


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

    def __init__(self, name=None, **kwargs):
        if name is not None:
            self.load(name)
        else:
            self.config = Config(**kwargs)

        self._setup_gen_params(self.config.gen_ks, self.config.gen_fn)
        self._setup_dis_params(self.config.dis_ks, self.config.dis_fn)

        if name is None:
            self.__sample_initials()
            self._setup_wave_params()

        self.__build_network()
        self.__build_obj()
        self.__compile_network()

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

    def __sample_initials(self):
        self.w_init = lasagne.init.Normal(std=0.02)
        self.b_init = lasagne.init.Constant(val=0.0)
        self.g_init = lasagne.init.Normal(mean=1., std=0.02)

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
            last = self.gen_fn[l]

        self.gen_W += [sharedX(self.w_init.sample(
            (last,self.gen_fn[-1], self.gen_ks[-1][0],self.gen_ks[-1][1])))]

    def _spatial_generator(self, inlayer, is_const=False):
        params = [self.wave_params, self.gen_W, self.gen_g, self.gen_b]
        if is_const:
            for i, p in enumerate(params):
                params[i] = [zero_grad(w) for w in p]
        wave_params, gen_W, gen_g, gen_b = params

        layers = [inlayer]
        layers.append(periodic(inlayer, self.config, wave_params))

        means = []
        inv_stds = []
        for l in range(self.gen_depth - 1):
            layers.append(batch_norm(
                tconv(layers[-1], self.gen_fn[l], self.gen_ks[l],
                      gen_W[l], nonlinearity=relu),
                gamma=gen_g[l], beta=gen_b[l], alpha=1.0))
            means += [layers[-1].input_layer.mean]
            inv_stds += [layers[-1].input_layer.inv_std]
        output  = tconv(layers[-1], self.gen_fn[-1], self.gen_ks[-1],
                        gen_W[-1], nonlinearity=tanh)
        if not hasattr(self, 'means'):
            self.means = means
            self.inv_stds = inv_stds

        return output

    def _spatial_generator_det(self, inlayer):
        layers = [inlayer]
        layers.append(periodic(inlayer, self.config, self.wave_params))

        for l in range(self.gen_depth - 1):
            layers.append(batch_norm(
                tconv(layers[-1], self.gen_fn[l], self.gen_ks[l],
                      self.gen_W[l], nonlinearity=relu),
                gamma=self.gen_g[l], beta=self.gen_b[l],
                mean=self.means[l], inv_std=self.inv_stds[l]))
        output = tconv(layers[-1], self.gen_fn[-1], self.gen_ks[-1],
                       self.gen_W[-1], nonlinearity=tanh)
        return output

    def __spatial_discriminator(self, inlayer):
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

    def __build_network(self):
        self.Z = lasagne.layers.InputLayer((None, self.config.nz, None, None))
        self.X = lasagne.layers.InputLayer((None, self.config.nc,
                                            self.config.npx, self.config.npx))
        self.gen_X = self._spatial_generator(self.Z)
        self.gen_X_det = self._spatial_generator_det(self.Z)

        if not isinstance(self, (InversePSGAN, InversePSGAN2)):
            self.d_real = self.__spatial_discriminator(self.X)
            self.d_fake = self.__spatial_discriminator(self.gen_X)

    def __build_obj(self):
        self.gen_X_out = get_output(self.gen_X)
        self.gen_X_det_out = get_output(self.gen_X_det, deterministic=True)

        if not isinstance(self, (InversePSGAN, InversePSGAN2)):
            d_real_out = get_output(self.d_real)
            d_fake_out = get_output(self.d_fake)

            params_g = get_all_params(self.gen_X, trainable=True)
            params_d = get_all_params(self.d_real, trainable=True)

            l2_g = regularize_network_params(self.gen_X,
                                             lasagne.regularization.l2)
            l2_d = regularize_network_params(self.d_real,
                                             lasagne.regularization.l2)
            self.obj_d = -T.mean(T.log(1-d_fake_out)) \
                         - T.mean(T.log(d_real_out)) \
                         + self.config.l2_fac * l2_d
            self.obj_g = -T.mean(T.log(d_fake_out)) + self.config.l2_fac * l2_g

            self.updates_d = lasagne.updates.adam(
                self.obj_d, params_d, self.config.lr, self.config.b1)
            self.updates_g = lasagne.updates.adam(
                self.obj_g, params_g, self.config.lr, self.config.b1)

    def __compile_network(self):
        if not isinstance(self, (InversePSGAN, InversePSGAN2)):
            logger = utils.create_logger('run_psgan.psgan_compile',
                                         stream=sys.stdout)
            logger.info("Compiling the network...")
            self.train_d = theano.function(
                [self.X.input_var, self.Z.input_var], self.obj_d,
                updates=self.updates_d, allow_input_downcast=True)
            logger.info("Discriminator done.")
            self.train_g = theano.function(
                [self.Z.input_var], self.obj_g, updates=self.updates_g,
                allow_input_downcast=True)
            logger.info("Generator done.")
            self.generate = theano.function(
                [self.Z.input_var], self.gen_X_out,
                allow_input_downcast=True)
            self.generate_det = theano.function(
                [self.Z.input_var], self.gen_X_det_out,
                allow_input_downcast=True)
            logger.info("generate function done.")

    def save(self, name):
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

        vals["means"] = [p.get_value() for p in self.means]
        vals["inv_stds"] = [p.get_value() for p in self.inv_stds]

        joblib.dump(vals, name, True)

    def load(self, name):
        logger = utils.create_logger('run_psgan.psgan_load', stream=sys.stdout)
        logger.info("loading parameters from file: {}".format(name))

        vals = joblib.load(name)
        self.config = vals["config"]

        self.dis_W = [sharedX(p) for p in vals["dis_W"]]
        self.dis_g = [sharedX(p) for p in vals["dis_g"]]
        self.dis_b = [sharedX(p) for p in vals["dis_b"]]

        self.gen_W = [sharedX(p) for p in vals["gen_W"]]
        self.gen_g = [sharedX(p) for p in vals["gen_g"]]
        self.gen_b = [sharedX(p) for p in vals["gen_b"]]

        self.wave_params = [sharedX(p) for p in vals["wave_params"]]

        self.means = [sharedX(p) for p in vals["means"]]
        self.inv_stds = [sharedX(p) for p in vals["inv_stds"]]

        self.config.gen_ks = []
        self.config.gen_fn = []
        l = len(vals["gen_W"])
        for i in range(l):
            if i == 0:
                self.config.nz = vals["gen_W"][i].shape[0]
            else:
                self.config.gen_fn += [vals["gen_W"][i].shape[0]]
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
        self.config.dis_fn = self.config.dis_fn[1:] + [1]


class InversePSGAN(PSGAN):

    def __init__(self, name=None, compile=True, **kwargs):
        super(InversePSGAN, self).__init__(name, **kwargs)

        self._setup_gen_z_params(self.config.gen_z_ks, self.config.gen_z_fn)

        if name is None:
            self._sample_initials()

        self._build_network()
        self._build_obj()
        if compile:
            self._compile_network()

    def _setup_gen_z_params(self, gen_z_ks, gen_z_fn):
        if gen_z_ks == None:
            self.gen_z_ks = [(5, 5)] * 5 + [(1, 1)]
        else:
            self.gen_z_ks = gen_z_ks

        self.gen_z_depth = len(self.gen_z_ks)

        if gen_z_fn != None:
            assert len(gen_z_fn) == len(
                self.gen_z_ks), 'Layer number of filter numbers and sizes does not match.'
            self.gen_z_fn = gen_z_fn
        else:
            self.gen_z_fn = [64] * self.gen_z_depth

    def _sample_initials(self):
        self.dis_W = []
        self.dis_b = []
        self.dis_g = []
        self.dis_W.append(
            (sharedX(self.w_init.sample((self.dis_fn[0], self.config.nc,
                                    self.dis_ks[0][0], self.dis_ks[0][1]))),
             sharedX(self.w_init.sample((self.dis_fn[0], 1, self.dis_ks[0][0],
                                    self.dis_ks[0][1])))))
        for l in range(self.dis_depth - 2):
            self.dis_W.append(sharedX(self.w_init.sample(
                (self.dis_fn[l + 1], 2 * self.dis_fn[l]
                if l == 0 else self.dis_fn[l], self.dis_ks[l + 1][0],
                 self.dis_ks[l + 1][1]))))
            self.dis_b.append(sharedX(self.b_init.sample((self.dis_fn[l + 1]))))
            self.dis_g.append(sharedX(self.g_init.sample((self.dis_fn[l + 1]))))
        self.dis_W.append(sharedX(self.w_init.sample(
            (self.dis_fn[-1], self.dis_fn[-2],
             self.dis_ks[-1][0], self.dis_ks[-1][1]))))

        self.gen_z_W = []
        self.gen_z_b = []
        self.gen_z_g = []
        self.gen_z_W.append(
            sharedX(self.w_init.sample(
                (self.gen_z_fn[0], self.config.nc, self.gen_z_ks[0][0],
                 self.gen_z_ks[0][1]))))
        for l in range(self.gen_z_depth - 1):
            self.gen_z_W.append(sharedX(self.w_init.sample(
                (self.gen_z_fn[l + 1], self.gen_z_fn[l],
                 self.gen_z_ks[l + 1][0], self.gen_z_ks[l + 1][1]))))
            self.gen_z_b.append(sharedX(self.b_init.sample(
                (self.gen_z_fn[l + 1]))))
            self.gen_z_g.append(sharedX(self.g_init.sample(
                (self.gen_z_fn[l + 1]))))

        self.transform_z_W = sharedX(self.w_init.sample((1, 1, 5, 5)))

    def _spatial_generator_Z(self, inlayer):
        layers = [inlayer]
        layers.append(conv(layers[-1], self.gen_z_fn[0], self.gen_z_ks[0],
                           self.gen_z_W[0], None, nonlinearity=Lrelu(0.2)))
        means_z = []
        inv_stds_z = []
        for l in range(1, self.gen_z_depth - 1):
            layers.append(batch_norm(
                conv(layers[-1], self.gen_z_fn[l], self.gen_z_ks[l],
                     self.gen_z_W[l], None, nonlinearity=Lrelu(0.2)),
                gamma=self.gen_z_g[l - 1], beta=self.gen_z_b[l - 1], alpha=1.0))
            means_z += [layers[-1].input_layer.mean]
            inv_stds_z += [layers[-1].input_layer.inv_std]
        output = conv(layers[-1], self.gen_z_fn[-1], self.gen_z_ks[-1],
                      self.gen_z_W[-1], None, nonlinearity=tanh, stride=1)
        output = lasagne.layers.Pool2DLayer(output, self.config.zx,
                                            mode='average_inc_pad')
        if not hasattr(self, 'means_z'):
            self.means_z = means_z
            self.inv_stds_z = inv_stds_z

        return output

    def _spatial_generator_Z_det(self, inlayer):
        layers = [inlayer]
        layers.append(conv(layers[-1], self.gen_z_fn[0], self.gen_z_ks[0],
                           self.gen_z_W[0], None, nonlinearity=Lrelu(0.2)))
        for l in range(1, self.gen_z_depth - 1):
            layers.append(batch_norm(
                conv(layers[-1], self.gen_z_fn[l], self.gen_z_ks[l],
                     self.gen_z_W[l], None, nonlinearity=Lrelu(0.2)),
                gamma=self.gen_z_g[l - 1], beta=self.gen_z_b[l - 1],
                mean=self.means_z[l - 1], inv_std=self.inv_stds_z[l - 1]))
        output = conv(layers[-1], self.gen_z_fn[-1], self.gen_z_ks[-1],
                      self.gen_z_W[-1], None, nonlinearity=tanh, stride=1)
        output = lasagne.layers.Pool2DLayer(output, self.config.zx,
                                            mode='average_inc_pad')

        return output

    def _transform_Z(self, Z):
        def find_s_p(i_size, o_size):
            p = 0
            while (o_size - 5 + 2 * p) % (i_size - 1) != 0:
                p += 1
            s = (o_size - 5 + 2 * p) / (i_size - 1)
            return s, p

        s, p = find_s_p(self.config.nz_global, self.config.npx)
        Z = lasagne.layers.ReshapeLayer(Z, (-1, 1, self.config.nz_global, 1))
        Z = lasagne.layers.ConcatLayer([Z] * self.config.nz_global, axis=3)
        Z = tconv(Z, 1, 5, self.transform_z_W, tanh, stride=s, crop=p)
        return Z

    def _spatial_discriminator(self, X, Z):
        layers = []
        X = conv(X, self.dis_fn[0], self.dis_ks[0], self.dis_W[0][0], None,
                 nonlinearity=Lrelu(0.2))
        Z = conv(Z, self.dis_fn[0], self.dis_ks[0], self.dis_W[0][1], None,
                 nonlinearity=Lrelu(0.2))
        X_Z = lasagne.layers.ConcatLayer([X, Z], axis=1)
        layers.append(X_Z)
        for l in range(1, self.dis_depth - 1):
            layers.append(batch_norm(
                conv(layers[-1], self.dis_fn[l], self.dis_ks[l],
                     self.dis_W[l], None, nonlinearity=Lrelu(0.2)),
                gamma=self.dis_g[l - 1], beta=self.dis_b[l - 1]))
        output = conv(layers[-1], self.dis_fn[-1], self.dis_ks[-1],
                      self.dis_W[-1], None, nonlinearity=sigmoid)

        return output, X_Z

    def _build_network(self):
        self.Z_global = lasagne.layers.InputLayer((None, self.config.nz_global,
                                                   None, None))
        self.Z_loc_and_period = lasagne.layers.SliceLayer(
            self.Z, indices=slice(self.config.nz_global, None), axis=1)

        self.gen_Z = self._spatial_generator_Z(self.X)
        self.gen_Z_det = self._spatial_generator_Z_det(self.X)

        gen_X_const = self._spatial_generator(self.Z, is_const=True)
        self.Z_g_reconst = self._spatial_generator_Z(gen_X_const)

        self.gen_Z_upscaled = lasagne.layers.ConcatLayer(
            [self.gen_Z] * self.config.zx, axis=2)
        self.gen_Z_upscaled = lasagne.layers.ConcatLayer(
            [self.gen_Z_upscaled] * self.config.zx, axis=3)
        self.gen_Z_full = lasagne.layers.ConcatLayer(
            [self.gen_Z_upscaled, self.Z_loc_and_period], axis=1)
        self.X_reconst = self._spatial_generator(self.gen_Z_full, is_const=True)

        Z_transformed = self._transform_Z(self.Z_global)
        self.d_fake, X_Z_fake = self._spatial_discriminator(
            self.gen_X, Z_transformed)

        self.gen_Z_transformed = self._transform_Z(self.gen_Z)
        self.d_real, X_Z_real = self._spatial_discriminator(
            self.X, self.gen_Z_transformed)

    def _build_obj(self):
        self.Z_global_out = get_output(self.Z_global)
        self.gen_Z_out = get_output(self.gen_Z)
        self.gen_Z_det_out = get_output(self.gen_Z_det, deterministic=True)
        self.Z_g_reconst_out = get_output(self.Z_g_reconst)
        self.X_out = get_output(self.X)
        self.X_reconst_out = get_output(self.X_reconst)

        d_real_out = get_output(self.d_real)
        d_fake_out = get_output(self.d_fake)

        params_g = get_all_params(self.gen_X, trainable=True)
        params_g += get_all_params(self.gen_Z_transformed, trainable=True)
        params_d = list(self.dis_W[0]) + self.dis_W[1:] \
                   + self.dis_b + self.dis_g
        l2_g = regularize_network_params(self.gen_X,
                                         lasagne.regularization.l2)
        l2_g_z = regularize_network_params(self.gen_Z_transformed,
                                           lasagne.regularization.l2)
        l2_d = lasagne.regularization.apply_penalty(
            list(self.dis_W[0]) + self.dis_W[1:], lasagne.regularization.l2)
        z_reconst_loss = T.mean((self.Z_global_out - self.Z_g_reconst_out)**2)
        self.x_reconst_loss = T.mean((self.X_out - self.X_reconst_out)**2)

        self.obj_d = -T.mean(T.log(1 - d_fake_out)) \
                     - T.mean(T.log(d_real_out)) \
                     + self.config.l2_fac * l2_d
        self.obj_g = -T.mean(T.log(d_fake_out)) \
                     - T.mean(T.log(1 - d_real_out)) \
                     + self.config.l2_fac * l2_g + self.config.l2_fac * l2_g_z \
                     + self.config.z_reconst_fac * z_reconst_loss \
                     + self.config.x_reconst_fac * self.x_reconst_loss
        self.updates_d = lasagne.updates.adam(
            self.obj_d, params_d, self.config.lr, self.config.b1)
        self.updates_g = lasagne.updates.adam(
            self.obj_g, params_g, self.config.lr, self.config.b1)

    def _compile_network(self):
        logger = utils.create_logger('run_psgan.invpsgan_compile',
                                     stream=sys.stdout)
        logger.info("Compiling the network...")
        self.train_d = theano.function(
            [self.X.input_var, self.Z.input_var, self.Z_global.input_var],
            self.obj_d, updates=self.updates_d, allow_input_downcast=True)
        logger.info("Discriminator done.")
        self.train_g = theano.function(
            [self.X.input_var, self.Z.input_var, self.Z_global.input_var],
            self.obj_g, updates=self.updates_g, allow_input_downcast=True)
        logger.info("Generator done.")
        self.generate = theano.function(
            [self.Z.input_var], self.gen_X_out, allow_input_downcast=True)
        self.generate_det = theano.function(
            [self.Z.input_var], self.gen_X_det_out,
            allow_input_downcast=True)
        self.generate_z = theano.function(
            [self.X.input_var], self.gen_Z_out,
            allow_input_downcast=True)
        self.generate_z_det = theano.function(
            [self.X.input_var], self.gen_Z_det_out,
            allow_input_downcast=True)
        logger.info("generate function done.")

    def save(self, name):
        logger = logging.getLogger('run_psgan.invpsgan_save')
        logger.info("saving InversePSGAN parameters in file: {}".format(name))
        vals = {}
        vals["config"] = self.config
        vals["dis_W"] = [tuple([p.get_value() for p in self.dis_W[0]])]
        vals["dis_W"] += [p.get_value() for p in self.dis_W[1:]]
        vals["dis_g"] = [p.get_value() for p in self.dis_g]
        vals["dis_b"] = [p.get_value() for p in self.dis_b]

        vals["gen_W"] = [p.get_value() for p in self.gen_W]
        vals["gen_g"] = [p.get_value() for p in self.gen_g]
        vals["gen_b"] = [p.get_value() for p in self.gen_b]

        vals["gen_z_W"] = [p.get_value() for p in self.gen_z_W]
        vals["gen_z_g"] = [p.get_value() for p in self.gen_z_g]
        vals["gen_z_b"] = [p.get_value() for p in self.gen_z_b]
        vals["transform_z_W"] = [self.transform_z_W.get_value()]

        vals["wave_params"] = [p.get_value() for p in self.wave_params]

        vals["means"] = [p.get_value() for p in self.means]
        vals["inv_stds"] = [p.get_value() for p in self.inv_stds]

        vals["means_z"] = [p.get_value() for p in self.means_z]
        vals["inv_stds_z"] = [p.get_value() for p in self.inv_stds_z]

        joblib.dump(vals, name, True)

    def load(self, name):
        logger = utils.create_logger('run_psgan.invpsgan_load',
                                     stream=sys.stdout)
        logger.info("loading parameters from file: {}".format(name))

        vals = joblib.load(name)
        self.config = vals["config"]

        self.dis_W = [tuple([sharedX(p) for p in vals["dis_W"][0]])]
        self.dis_W += [sharedX(p) for p in vals["dis_W"][1:]]
        self.dis_g = [sharedX(p) for p in vals["dis_g"]]
        self.dis_b = [sharedX(p) for p in vals["dis_b"]]

        self.gen_W = [sharedX(p) for p in vals["gen_W"]]
        self.gen_g = [sharedX(p) for p in vals["gen_g"]]
        self.gen_b = [sharedX(p) for p in vals["gen_b"]]

        self.gen_z_W = [sharedX(p) for p in vals["gen_z_W"]]
        self.gen_z_g = [sharedX(p) for p in vals["gen_z_g"]]
        self.gen_z_b = [sharedX(p) for p in vals["gen_z_b"]]
        self.transform_z_W = sharedX(vals["transform_z_W"][0])

        self.wave_params = [sharedX(p) for p in vals["wave_params"]]

        self.means = [sharedX(p) for p in vals["means"]]
        self.inv_stds = [sharedX(p) for p in vals["inv_stds"]]

        self.means_z = [sharedX(p) for p in vals["means_z"]]
        self.inv_stds_z = [sharedX(p) for p in vals["inv_stds_z"]]

        self.config.gen_ks = []
        self.config.gen_fn = []
        l = len(vals["gen_W"])
        for i in range(l):
            if i == 0:
                self.config.nz = vals["gen_W"][i].shape[0]
            else:
                self.config.gen_fn += [vals["gen_W"][i].shape[0]]
            self.config.gen_ks += [(vals["gen_W"][i].shape[2],
                                    vals["gen_W"][i].shape[3])]
        self.config.nc = vals["gen_W"][i].shape[1]
        self.config.gen_fn += [self.config.nc]

        # self.config.dis_ks = []
        # self.config.dis_fn = []
        # l = len(vals["dis_W"])
        # for i in range(l):
        #     if i == 0:
        #         pass
        #     else:
        #         self.config.dis_fn += [vals["dis_W"][i].shape[1]]
        #     self.config.dis_ks += [(vals["gen_W"][i].shape[2],
        #                             vals["gen_W"][i].shape[3])]
        # self.config.dis_fn += [1]


class InversePSGAN2(PSGAN):

    def __init__(self, name=None, compile=True, **kwargs):
        super(InversePSGAN2, self).__init__(name, **kwargs)

        self._setup_gen_z_params(self.config.gen_z_ks, self.config.gen_z_fn)

        if name is None:
            self._sample_initials()

        self._build_network()
        self._build_obj()
        if compile:
            self._compile_network()

    def _setup_gen_z_params(self, gen_z_ks, gen_z_fn):
        if gen_z_ks == None:
            self.gen_z_ks = [(5, 5)] * 5 + [(1, 1)]
        else:
            self.gen_z_ks = gen_z_ks

        self.gen_z_depth = len(self.gen_z_ks)

        if gen_z_fn != None:
            assert len(gen_z_fn) == len(
                self.gen_z_ks), 'Layer number of filter numbers and sizes does not match.'
            self.gen_z_fn = gen_z_fn
        else:
            self.gen_z_fn = [64] * self.gen_z_depth

    def _sample_initials(self):
        self.gen_z_W = []
        self.gen_z_b = []
        self.gen_z_g = []
        self.gen_z_W.append(
            sharedX(self.w_init.sample(
                (self.gen_z_fn[0], self.config.nc, self.gen_z_ks[0][0],
                 self.gen_z_ks[0][1]))))
        for l in range(self.gen_z_depth - 1):
            self.gen_z_W.append(sharedX(self.w_init.sample(
                (self.gen_z_fn[l + 1], self.gen_z_fn[l],
                 self.gen_z_ks[l + 1][0], self.gen_z_ks[l + 1][1]))))
            self.gen_z_b.append(sharedX(self.b_init.sample(
                (self.gen_z_fn[l + 1]))))
            self.gen_z_g.append(sharedX(self.g_init.sample(
                (self.gen_z_fn[l + 1]))))

    def _spatial_generator_Z(self, inlayer):
        layers = [inlayer]
        layers.append(conv(layers[-1], self.gen_z_fn[0], self.gen_z_ks[0],
                           self.gen_z_W[0], None, nonlinearity=Lrelu(0.2)))
        means_z = []
        inv_stds_z = []
        for l in range(1, self.gen_z_depth - 1):
            layers.append(batch_norm(
                conv(layers[-1], self.gen_z_fn[l], self.gen_z_ks[l],
                     self.gen_z_W[l], None, nonlinearity=Lrelu(0.2)),
                gamma=self.gen_z_g[l - 1], beta=self.gen_z_b[l - 1], alpha=1.0))
            means_z += [layers[-1].input_layer.mean]
            inv_stds_z += [layers[-1].input_layer.inv_std]
        output = conv(layers[-1], self.gen_z_fn[-1], self.gen_z_ks[-1],
                      self.gen_z_W[-1], None, nonlinearity=tanh, stride=1)
        output = lasagne.layers.Pool2DLayer(output, self.config.zx,
                                            mode='average_inc_pad')
        if not hasattr(self, 'means_z'):
            self.means_z = means_z
            self.inv_stds_z = inv_stds_z

        return output

    def _spatial_generator_Z_det(self, inlayer):
        layers = [inlayer]
        layers.append(conv(layers[-1], self.gen_z_fn[0], self.gen_z_ks[0],
                           self.gen_z_W[0], None, nonlinearity=Lrelu(0.2)))
        for l in range(1, self.gen_z_depth - 1):
            layers.append(batch_norm(
                conv(layers[-1], self.gen_z_fn[l], self.gen_z_ks[l],
                     self.gen_z_W[l], None, nonlinearity=Lrelu(0.2)),
                gamma=self.gen_z_g[l - 1], beta=self.gen_z_b[l - 1],
                mean=self.means_z[l - 1], inv_std=self.inv_stds_z[l - 1]))
        output = conv(layers[-1], self.gen_z_fn[-1], self.gen_z_ks[-1],
                      self.gen_z_W[-1], None, nonlinearity=tanh, stride=1)
        output = lasagne.layers.Pool2DLayer(output, self.config.zx,
                                            mode='average_inc_pad')

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

    def _build_network(self):
        self.Z_loc_and_period = lasagne.layers.SliceLayer(
            self.Z, indices=slice(self.config.nz_global, None), axis=1)

        self.gen_Z = self._spatial_generator_Z(self.X)
        self.gen_Z_det = self._spatial_generator_Z_det(self.X)

        self.gen_Z_upscaled = lasagne.layers.ConcatLayer(
            [self.gen_Z] * self.config.zx, axis=2)
        self.gen_Z_upscaled = lasagne.layers.ConcatLayer(
            [self.gen_Z_upscaled] * self.config.zx, axis=3)
        self.gen_Z_full = lasagne.layers.ConcatLayer(
            [self.gen_Z_upscaled, self.Z_loc_and_period], axis=1)
        self.X_reconst = self._spatial_generator(self.gen_Z_full, is_const=True)

        self.X_double = lasagne.layers.ConcatLayer([self.X] * 2, axis=3)
        self.d_real = self._spatial_discriminator(self.X_double)

        self.gen_X_double = lasagne.layers.ConcatLayer(
            [self.gen_X, self.X_reconst], axis=3)
        self.d_fake = self._spatial_discriminator(self.gen_X_double)

    def _build_obj(self):
        self.gen_Z_out = get_output(self.gen_Z)
        self.gen_Z_det_out = get_output(self.gen_Z_det, deterministic=True)
        self.X_out = get_output(self.X)
        self.X_reconst_out = get_output(self.X_reconst)

        d_real_out = get_output(self.d_real)
        d_fake_out = get_output(self.d_fake)

        params_g = get_all_params(self.gen_X, trainable=True)
        params_g += get_all_params(self.gen_Z, trainable=True)
        params_d = get_all_params(self.d_real, trainable=True)
        l2_g = regularize_network_params(self.gen_X,
                                         lasagne.regularization.l2)
        l2_g_z = regularize_network_params(self.gen_Z,
                                           lasagne.regularization.l2)
        l2_d = regularize_network_params(self.d_real,
                                         lasagne.regularization.l2)

        self.obj_d = -T.mean(T.log(1 - d_fake_out)) \
                     - T.mean(T.log(d_real_out)) \
                     + self.config.l2_fac * l2_d
        self.obj_g = -T.mean(T.log(d_fake_out)) \
                     + self.config.l2_fac * l2_g + self.config.l2_fac * l2_g_z
        self.updates_d = lasagne.updates.adam(
            self.obj_d, params_d, self.config.lr, self.config.b1)
        self.updates_g = lasagne.updates.adam(
            self.obj_g, params_g, self.config.lr, self.config.b1)

    def _compile_network(self):
        logger = utils.create_logger('run_psgan.invpsgan_compile',
                                     stream=sys.stdout)
        logger.info("Compiling the network...")
        self.train_d = theano.function(
            [self.X.input_var, self.Z.input_var],
            self.obj_d, updates=self.updates_d, allow_input_downcast=True)
        logger.info("Discriminator done.")
        self.train_g = theano.function(
            [self.X.input_var, self.Z.input_var],
            self.obj_g, updates=self.updates_g, allow_input_downcast=True)
        logger.info("Generator done.")
        self.generate = theano.function(
            [self.Z.input_var], self.gen_X_out, allow_input_downcast=True)
        self.generate_det = theano.function(
            [self.Z.input_var], self.gen_X_det_out,
            allow_input_downcast=True)
        self.generate_z = theano.function(
            [self.X.input_var], self.gen_Z_out,
            allow_input_downcast=True)
        self.generate_z_det = theano.function(
            [self.X.input_var], self.gen_Z_det_out,
            allow_input_downcast=True)
        logger.info("generate function done.")

    def save(self, name):
        logger = logging.getLogger('run_psgan.invpsgan_save')
        logger.info("saving InversePSGAN parameters in file: {}".format(name))
        vals = {}
        vals["config"] = self.config
        vals["dis_W"] = [p.get_value() for p in self.dis_W]
        vals["dis_g"] = [p.get_value() for p in self.dis_g]
        vals["dis_b"] = [p.get_value() for p in self.dis_b]

        vals["gen_W"] = [p.get_value() for p in self.gen_W]
        vals["gen_g"] = [p.get_value() for p in self.gen_g]
        vals["gen_b"] = [p.get_value() for p in self.gen_b]

        vals["gen_z_W"] = [p.get_value() for p in self.gen_z_W]
        vals["gen_z_g"] = [p.get_value() for p in self.gen_z_g]
        vals["gen_z_b"] = [p.get_value() for p in self.gen_z_b]

        vals["wave_params"] = [p.get_value() for p in self.wave_params]

        vals["means"] = [p.get_value() for p in self.means]
        vals["inv_stds"] = [p.get_value() for p in self.inv_stds]

        vals["means_z"] = [p.get_value() for p in self.means_z]
        vals["inv_stds_z"] = [p.get_value() for p in self.inv_stds_z]

        joblib.dump(vals, name, True)

    def load(self, name):
        logger = utils.create_logger('run_psgan.invpsgan_load',
                                     stream=sys.stdout)
        logger.info("loading parameters from file: {}".format(name))

        vals = joblib.load(name)
        self.config = vals["config"]

        self.dis_W = [sharedX(p) for p in vals["dis_W"]]
        self.dis_g = [sharedX(p) for p in vals["dis_g"]]
        self.dis_b = [sharedX(p) for p in vals["dis_b"]]

        self.gen_W = [sharedX(p) for p in vals["gen_W"]]
        self.gen_g = [sharedX(p) for p in vals["gen_g"]]
        self.gen_b = [sharedX(p) for p in vals["gen_b"]]

        if hasattr(vals, 'gen_z_W'):
            self.gen_z_W = [sharedX(p) for p in vals["gen_z_W"]]
            self.gen_z_g = [sharedX(p) for p in vals["gen_z_g"]]
            self.gen_z_b = [sharedX(p) for p in vals["gen_z_b"]]
        else:
            self._sample_initials()

        self.wave_params = [sharedX(p) for p in vals["wave_params"]]

        self.means = [sharedX(p) for p in vals["means"]]
        self.inv_stds = [sharedX(p) for p in vals["inv_stds"]]

        if hasattr(vals, 'means_z'):
            self.means_z = [sharedX(p) for p in vals["means_z"]]
            self.inv_stds_z = [sharedX(p) for p in vals["inv_stds_z"]]

        self.config.gen_ks = []
        self.config.gen_fn = []
        l = len(vals["gen_W"])
        for i in range(l):
            if i == 0:
                self.config.nz = vals["gen_W"][i].shape[0]
            else:
                self.config.gen_fn += [vals["gen_W"][i].shape[0]]
            self.config.gen_ks += [(vals["gen_W"][i].shape[2],
                                    vals["gen_W"][i].shape[3])]
        self.config.nc = vals["gen_W"][i].shape[1]
        self.config.gen_fn += [self.config.nc]

