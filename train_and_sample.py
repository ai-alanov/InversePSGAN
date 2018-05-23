import numpy as np
import os
import sys
from tqdm import tqdm

import utils

np.random.seed(1234)


def train(model, config, logger, options, model_dir, samples_dir):
    utils.makedirs(samples_dir)

    for epoch in tqdm(range(options.n_epochs), file=sys.stdout):
        logger.info("Epoch {}".format(epoch))
        Gcost = []
        Dcost = []

        for it in tqdm(range(options.n_iters), file=sys.stdout):
            Znp = utils.sample_noise_tensor(config, options.b_size, config.zx)

            if it % (config.k + 1) == 0:
                Gcost.append(model.train_g(Znp))
            else:
                samples = next(config.data_iter(options.t_path, options.b_size))
                Dcost.append(model.train_d(samples, Znp))
        msg = "Gcost = {}, Dcost = {}"
        logger.info(msg.format(np.mean(Gcost), np.mean(Dcost)))

        samples = next(config.data_iter(options.b_size))
        samples = np.concatenate(samples, axis=2)

        Znp = utils.sample_noise_tensor(config, options.b_size, config.zx)
        gen_samples = model.generate(Znp)
        gen_samples = np.concatenate(gen_samples, axis=2)

        z_sample = utils.sample_noise_tensor(config, 1, config.zx_sample,
                                             config.zx_sample_quilt)
        large_sample = model.generate(z_sample)[0]

        utils.save_samples(samples_dir, [samples, gen_samples, large_sample],
                           ['real', 'gen', 'large'], epoch=epoch)
        if (epoch+1) % 10 == 0:
            model_file = 'epoch_{}.model'.format(epoch)
            model.save(os.path.join(model_dir, model_file))


def sample(model, config, samples_dir, n_samples=5):
    utils.makedirs(samples_dir)

    all_samples = []
    for i in range(n_samples):
        global_noise = np.random.uniform(-1., 1., (1, config.nz_global, 1, 1))
        z_samples = utils.sample_noise_tensor(config, 5, config.zx,
                                              global_noise=global_noise)
        gen_samples = model.generate_det(z_samples)
        gen_samples = np.concatenate(gen_samples, axis=2)
        all_samples.append(gen_samples)
    utils.save_samples(samples_dir, all_samples,
                       [str(i+1) for i in range(n_samples)])