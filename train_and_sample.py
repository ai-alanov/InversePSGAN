import numpy as np
import os
import sys
from tqdm import tqdm

import utils

np.random.seed(1234)


def train(model, config, logger, options, model_dir, samples_dir, inverse=False):
    utils.makedirs(samples_dir)

    for epoch in tqdm(range(options.n_epochs), file=sys.stdout):
        logger.info("Epoch {}".format(epoch))
        Gcost = []
        Dcost = []

        samples_generator = config.data_iter(options.t_path, options.b_size)

        for it in tqdm(range(options.n_iters), file=sys.stdout):
            Z_global = None
            if inverse:
                Z_global = np.random.uniform(
                    -1., 1., (options.b_size, config.nz_global, 1, 1))
            Z_samples = utils.sample_noise_tensor(
                config, options.b_size, config.zx, global_noise=Z_global)

            X_samples = next(samples_generator)
            if it % (config.k + 1) == 0:
                if not inverse:
                    Gcost.append(model.train_g(Z_samples))
                else:
                    Gcost.append(model.train_g(X_samples, Z_samples, Z_global))
            else:
                if not inverse:
                    Dcost.append(model.train_d(X_samples, Z_samples))
                else:
                    Dcost.append(model.train_d(X_samples, Z_samples, Z_global))
        msg = "Gcost = {}, Dcost = {}"
        logger.info(msg.format(np.mean(Gcost), np.mean(Dcost)))

        X_samples = next(samples_generator)
        X_samples = np.concatenate(X_samples, axis=2)

        Z_samples = utils.sample_noise_tensor(config, options.b_size, config.zx)
        gen_samples = model.generate(Z_samples)
        gen_samples = np.concatenate(gen_samples, axis=2)

        z_sample = utils.sample_noise_tensor(config, 1, config.zx_sample,
                                             config.zx_sample_quilt)
        large_sample = model.generate(z_sample)[0]

        utils.save_samples(samples_dir, [X_samples, gen_samples, large_sample],
                           ['real', 'gen', 'large'], epoch=epoch)
        if (epoch+1) % 10 == 0:
            model_file = 'epoch_{}.model'.format(epoch)
            model.save(os.path.join(model_dir, model_file))


def sample(model, config, samples_dir, texture_path,
           n_samples=5, inverse=False):
    utils.makedirs(samples_dir)

    imgs = None
    if inverse:
        imgs = sorted(os.listdir(texture_path))[:n_samples]
    all_samples = []
    for i in range(n_samples):
        if inverse:
            global_noise = model.generate_z_det(imgs[i])
        else:
            global_noise = np.random.uniform(
                -1., 1., (1, config.nz_global, 1, 1))
        z_samples = utils.sample_noise_tensor(config, 5, config.zx,
                                              global_noise=global_noise)
        gen_samples = []
        if inverse:
            gen_samples += [imgs[i]]
        gen_samples += model.generate_det(z_samples)
        gen_samples = np.concatenate(gen_samples, axis=2)
        all_samples.append(gen_samples)
    utils.save_samples(samples_dir, all_samples,
                       [str(i+1) for i in range(n_samples)])