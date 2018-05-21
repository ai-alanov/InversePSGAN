import numpy as np
from tqdm import tqdm
import logging
from optparse import OptionParser
import os
import sys
import glob
from datetime import datetime
import csv

from psgan import PSGAN
from data_io import save_tensor


def sample_noise_tensor(config, batch_size, zx, zx_qlt=None):
    Z = np.zeros((batch_size, config.nz, zx, zx))
    Z[:, config.nz_global:config.nz_global+config.nz_local] = \
        np.random.uniform(-1., 1., (batch_size, config.nz_local, zx, zx))

    if zx_qlt is None:
        Z[:, :config.nz_global] = \
            np.random.uniform(-1., 1., (batch_size, config.nz_global, 1, 1))
    else:
        Z_g = Z[:, :config.nz_global]
        for i in range(zx / zx_qlt):
            for j in range(zx / zx_qlt):
                Z_g[:, :, i*zx_qlt:(i+1)*zx_qlt, j*zx_qlt:(j+1)*zx_qlt] = \
                    np.random.uniform(-1., 1.,
                                      (batch_size, config.nz_global, 1, 1))

    if config.nz_periodic > 0:
        for i, pixel in zip(range(1, config.nz_periodic + 1),
                            np.linspace(30, 130, config.nz_periodic)):
            band = np.pi * (0.5 * i / float(config.nz_periodic) + 0.5)
            for h in range(zx):
                Z[:, -i * 2, :, h] = h * band
            for w in range(zx):
                Z[:, -i * 2 + 1, w] = w * band
    return Z


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_logging_file(log_dir, options):
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d"))
    n_files = len(glob.glob(os.path.join(log_dir, '*')))
    file_id = '{:06d}'.format(n_files if n_files else 1)
    log_file = file_id + datetime.now().strftime("_%H:%M:%S") + '.txt'
    log_file = os.path.join(log_dir, log_file)
    if os.path.exists(log_dir):
        with open(os.path.join(log_dir, 'id_to_options.csv'), 'a') as f:
            w = csv.DictWriter(f, ['id'] + sorted(options.keys()),
                               delimiter='\t')
            options['id'] = file_id
            w.writerow(options)
    else:
        makedirs(log_dir)
        with open(os.path.join(log_dir, 'id_to_options.csv'), 'w') as f:
            w = csv.DictWriter(f, ['id'] + sorted(options.keys()),
                               delimiter='\t')
            w.writeheader()
            options['id'] = file_id
            w.writerow(options)
    return log_file


def create_model_folder(model_dir, options):
    model_dir = os.path.join(model_dir, datetime.now().strftime("%Y-%m-%d"))
    n_folders = len(glob.glob(os.path.join(model_dir, '*')))
    folder_id = '{:06d}'.format(n_folders if n_folders else 1)
    model_folder = folder_id + datetime.now().strftime("_%H:%M:%S")
    model_folder = os.path.join(model_dir, model_folder)
    if os.path.exists(model_dir):
        with open(os.path.join(model_dir, 'id_to_options.csv'), 'a') as f:
            w = csv.DictWriter(f, ['id'] + sorted(options.keys()),
                               delimiter='\t')
            options['id'] = folder_id
            w.writerow(options)
    else:
        makedirs(model_dir)
        with open(os.path.join(model_dir, 'id_to_options.csv'), 'w') as f:
            w = csv.DictWriter(f, ['id'] + sorted(options.keys()),
                               delimiter='\t')
            w.writeheader()
            options['id'] = folder_id
            w.writerow(options)
    makedirs(model_folder)
    return model_folder


def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--mode", type='string', default='train',
                      help="train or sample")
    parser.add_option("--checkpoint", type='string', default='last',
                      help="load a model from checkpoint")
    parser.add_option("--data", type='string', default='texture',
                      help="path to data for training")
    (options, args) = parser.parse_args()

    log_file = create_logging_file('logs', vars(options))
    logger = logging.getLogger('run_psgan')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    log_format = '%(levelname)s:%(asctime)s:%(name)s: %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d:%H-%M-%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    class StreamToLogger(object):

        def __init__(self, stderr, logger):
            self.stderr = stderr
            self.logger = logger

        def write(self, buf):
            self.stderr.write(buf)
            for line in buf.rstrip().splitlines():
                self.logger.error(line.rstrip())

    stderr_logger = logging.getLogger('STDERR')
    stderr_handler = logging.FileHandler(log_file)
    stderr_handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    stderr_logger.addHandler(stderr_handler)
    sys.stderr = StreamToLogger(sys.stderr, stderr_logger)

    psgan = PSGAN()
    c = psgan.config
    c.print_info()
    z_sample = sample_noise_tensor(c, 1, c.zx_sample, c.zx_sample_quilt)
    epoch = 0
    tot_iter = 0

    samples_folder = log_file[:-len('%H:%M:%S.txt')] + 'samples'
    makedirs(samples_folder)
    model_folder = create_model_folder('models', vars(options))

    while epoch < c.epoch_count:
        epoch += 1
        logger.info("Epoch %d" % epoch)

        Gcost = []
        Dcost = []

        iters = c.epoch_iters / c.batch_size
        for it, samples in enumerate(tqdm(c.data_iter(), total=iters,
                                          file=sys.stdout)):
            if it >= iters:
                break
            tot_iter += 1

            Znp = sample_noise_tensor(c, c.batch_size, c.zx)

            if tot_iter % (c.k + 1) == 0:
                cost = psgan.train_g(Znp)
                Gcost.append(cost)
            else:
                cost = psgan.train_d(samples, Znp)
                Dcost.append(cost)
        msg = "Gcost = {}, Dcost = {}"
        logger.info(msg.format(np.mean(Gcost), np.mean(Dcost)))

        samples_epoch_folder = os.path.join(samples_folder, str(epoch))
        makedirs(samples_epoch_folder)
        slist = []
        for img in samples:
            slist += [img]
        img = np.concatenate(slist, axis=2)
        real_imgs_file = 'real_{}_epoch{}.jpg'.format(c.save_name, epoch)
        real_imgs_file = os.path.join(samples_epoch_folder, real_imgs_file)
        save_tensor(img, real_imgs_file)

        samples = psgan.generate(Znp)
        slist = []
        for img in samples:
            slist += [img]
        img = np.concatenate(slist, axis=2)
        gen_imgs_file = 'gen_{}_epoch{}.jpg'.format(c.save_name, epoch)
        gen_imgs_file = os.path.join(samples_epoch_folder, gen_imgs_file)
        save_tensor(img, gen_imgs_file)

        data = psgan.generate(z_sample)

        large_img_file = 'large{}_epoch{}.jpg'.format(c.save_name, epoch)
        large_img_file = os.path.join(samples_epoch_folder, large_img_file)
        save_tensor(data[0], large_img_file)
        model_file = '{}_epoch{}.psgan'.format(c.save_name, epoch)
        psgan.save(os.path.join(model_folder, model_file))


if __name__ == '__main__':
    main()