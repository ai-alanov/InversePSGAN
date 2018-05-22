import os
import glob
from datetime import datetime
import csv
import logging
import numpy as np

from data_io import save_tensor


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_logging_file(log_dir, options):
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d"))
    n_files = len(glob.glob(os.path.join(log_dir, '*')))
    file_id = '{:06d}'.format(n_files if n_files else 1)
    log_file = file_id + datetime.now().strftime("_%H:%M:%S")
    log_file = os.path.join(log_dir, log_file, 'log.txt')
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
    makedirs(os.path.dirname(log_file))
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


def find_checkpoint(model_dir, checkpoint):
    if not checkpoint:
        return None
    if checkpoint == 'last':
        print glob.glob(model_dir)
        print glob.glob(sorted(glob.glob(model_dir))[-1])
        model_folder = sorted(glob.glob(sorted(glob.glob(model_dir))[-1]))[-2]
        last_model_name = sorted(glob.glob(model_folder))[-1]
        return last_model_name


def create_logger(name, file=None, stream=None, level=logging.INFO, need_fmt=False,
                  fmt='%(levelname)s:%(asctime)s:%(name)s: %(message)s',
                  datefmt='%Y-%m-%d:%H-%M-%S'):
    logger = logging.getLogger(name)
    if file or stream:
        file_handler = logging.FileHandler(file) if file \
            else logging.StreamHandler(stream)
        file_handler.setLevel(level)
        if need_fmt:
            formatter = logging.Formatter(fmt, datefmt=datefmt)
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def copy_stream_to_log(stream, stream_name, file):
    class StreamToLogger(object):

        def __init__(self, stream, logger):
            self.stream = stream
            self.logger = logger

        def write(self, buffer):
            self.stream.write(buffer)
            for line in buffer.rstrip().splitlines():
                self.logger.error(line.rstrip())

    logger = logging.getLogger(stream_name)
    handler = logging.FileHandler(file)
    handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    logger.addHandler(handler)
    return StreamToLogger(stream, logger)


def save_samples(save_dir, epoch, real_samples, gen_samples, large_sample):
    epoch_dir = os.path.join(save_dir, 'epoch_{}'.format(epoch))
    makedirs(epoch_dir)

    samples = [np.concatenate(real_samples, axis=2),
               np.concatenate(gen_samples, axis=2), large_sample]
    samples_name = ['real', 'gen', 'large']
    for name, sample in zip(samples_name, samples):
        sample_file = '{}_epoch_{}.jpg'.format(name, epoch)
        sample_file = os.path.join(epoch_dir, sample_file)
        save_tensor(sample, sample_file)


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

