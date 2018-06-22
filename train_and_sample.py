import numpy as np
import os
import sys
from tqdm import tqdm
from collections import defaultdict

import utils
from data_io import get_images, get_random_patch

np.random.seed(1234)


def train(model, config, logger, options, model_dir, samples_dir,
          inverse=False, save_step=10):
    utils.makedirs(samples_dir)

    losses = defaultdict(list)
    for epoch in tqdm(range(options.n_epochs), file=sys.stdout):
        logger.info("Epoch {}".format(epoch))

        samples_generator = config.data_iter(options.data, options.b_size,
                                             inverse=inverse)

        for it in tqdm(range(options.n_iters), file=sys.stdout):
            Z_global = None
            if inverse:
                Z_global = np.random.uniform(
                    -1., 1., (options.b_size, config.nz_global, 1, 1))
            Z_samples = utils.sample_noise_tensor(
                config, options.b_size, config.zx, global_noise=Z_global)

            X_samples = next(samples_generator)
            if it % (config.k + 1) != 0:
                if inverse == 0:
                    losses['G_iter'].append(model.train_g(Z_samples))
                elif inverse == 1:
                    losses['G_iter'].append(model.train_g(X_samples, Z_samples,
                                                          Z_global))
                elif inverse >= 2:
                    losses['G_iter'].append(model.train_g(X_samples[0],
                                                          Z_samples))
            else:
                if inverse == 0:
                    losses['D_iter'].append(model.train_d(X_samples, Z_samples))
                elif inverse == 1:
                    losses['D_iter'].append(model.train_d(X_samples, Z_samples,
                                                          Z_global))
                elif inverse >= 2:
                    losses['D_iter'].append(model.train_d(
                        X_samples[0], X_samples[1], Z_samples))
        msg = "Gloss = {}, Dloss = {}"
        losses['G_epoch'].append(np.mean(losses['G_iter'][-options.n_iters:]))
        losses['D_epoch'].append(np.mean(losses['D_iter'][-options.n_iters:]))
        logger.info(msg.format(losses['G_epoch'][-1], losses['D_epoch'][-1]))

        X = next(samples_generator)
        real_samples, gen_samples, large_sample = utils.sample_after_iteration(
            model, X, inverse, config, options.b_size)
        utils.save_samples(
            samples_dir, [real_samples, gen_samples, large_sample],
            ['real', 'gen', 'large'], epoch=epoch)
        utils.save_plots(samples_dir, losses, epoch, options.n_iters)

        if (epoch+1) % save_step == 0:
            model_file = 'epoch_{:04d}.model'.format(epoch)
            model.save(os.path.join(model_dir, model_file))


def sample(model, config, samples_dir, texture_path,
           n_samples=20, n_z_samples=5, inverse=False):
    utils.makedirs(samples_dir)

    imgs = None
    if inverse:
        img_files = sorted(os.listdir(texture_path))[:n_samples]
        img_files = [texture_path + file for file in img_files]
        imgs = get_images(img_files)
        imgs = [get_random_patch(img, config.npx) for img in imgs]
        imgs = [np.reshape(img, (1,) + img.shape) for img in imgs]
    all_samples = []
    if inverse:
        X = np.concatenate(imgs, axis=0)
        global_noise = model.generate_z(X)
        global_noise = model.generate_z_det(X)
        z_samples = utils.sample_noise_tensor(config, n_z_samples, config.zx,
                                              global_noise=global_noise,
                                              per_each=True)
        gen_samples = model.generate(z_samples)
        gen_samples = model.generate_det(z_samples)
        all_samples = [[img, list(gen_samples[n_z_samples*i:n_z_samples*(i+1)])]
                       for i, img in enumerate(imgs)]
        all_samples = [np.concatenate(np.concatenate(samples, axis=0), axis=2)
                       for samples in all_samples]
        all_samples = [np.concatenate(all_samples, axis=1)]
        utils.save_samples(samples_dir, all_samples, ['inv_gens'])
        all_samples = []
    for i in range(n_samples):
        global_noise = np.random.uniform(-1., 1., (1, config.nz_global, 1, 1))
        z_samples = utils.sample_noise_tensor(config, n_z_samples, config.zx,
                                              global_noise=global_noise)
        gen_samples = model.generate(z_samples)
        gen_samples = np.concatenate(gen_samples, axis=2)
        all_samples.append(gen_samples)
    all_samples = [np.concatenate(all_samples, axis=1)]
    utils.save_samples(samples_dir, all_samples, ['gens'])
