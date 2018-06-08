import numpy as np
import os
import sys
from tqdm import tqdm

import utils
from data_io import get_images

np.random.seed(1234)


def train(model, config, logger, options, model_dir, samples_dir,
          inverse=False, save_step=10):
    utils.makedirs(samples_dir)

    for epoch in tqdm(range(options.n_epochs), file=sys.stdout):
        logger.info("Epoch {}".format(epoch))
        Gcost = []
        Dcost = []

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
                    Gcost.append(model.train_g(Z_samples))
                elif inverse == 1:
                    Gcost.append(model.train_g(X_samples, Z_samples, Z_global))
                elif inverse == 2:
                    Gcost.append(model.train_g(X_samples[0],
                                               Z_samples))
            else:
                if inverse == 0:
                    Dcost.append(model.train_d(X_samples, Z_samples))
                elif inverse == 1:
                    Dcost.append(model.train_d(X_samples, Z_samples, Z_global))
                elif inverse == 2:
                    Dcost.append(model.train_d(X_samples[0], X_samples[1],
                                               Z_samples))
        msg = "Gcost = {}, Dcost = {}"
        logger.info(msg.format(np.mean(Gcost), np.mean(Dcost)))

        X = next(samples_generator)
        if inverse == 2:
            X_samples = model.generate_x_double(X[0], X[1])
            X_samples = np.concatenate(X_samples, axis=1)
        else:
            X_samples = np.concatenate(X, axis=2)

        Z_samples = utils.sample_noise_tensor(config, options.b_size, config.zx)
        if inverse == 2:
            gen_samples = model.generate_gen_x_double(X[0], Z_samples)
            gen_samples = np.concatenate(gen_samples, axis=1)
        else:
            gen_samples = model.generate(Z_samples)
            gen_samples = np.concatenate(gen_samples, axis=2)

        z_sample = utils.sample_noise_tensor(config, 1, config.zx_sample,
                                             config.zx_sample_quilt)
        large_sample = model.generate(z_sample)[0]

        utils.save_samples(samples_dir, [X_samples, gen_samples, large_sample],
                           ['real', 'gen', 'large'], epoch=epoch)
        if (epoch+1) % save_step == 0:
            model_file = 'epoch_{:04d}.model'.format(epoch)
            model.save(os.path.join(model_dir, model_file))


def sample(model, config, samples_dir, texture_path,
           n_samples=20, inverse=False):
    utils.makedirs(samples_dir)

    imgs = None
    if inverse:
        img_files = sorted(os.listdir(texture_path))[:n_samples]
        img_files = [texture_path + file for file in img_files]
        imgs = get_images(img_files)
        imgs = [np.reshape(img, (1,) + img.shape) for img in imgs]
        cropped_imgs = []
        npx = config.npx
        for img in imgs:
            h = np.random.randint(img.shape[2] - npx)
            w = np.random.randint(img.shape[3] - npx)
            cropped_img = img[:, :, h:h + npx, w:w + npx]
            cropped_imgs.append(cropped_img)
        imgs = cropped_imgs
    all_samples = []
    if inverse:
        for i in range(n_samples):
            global_noise = model.generate_z_det(imgs[i])
            z_samples = utils.sample_noise_tensor(config, 5, config.zx,
                                                  global_noise=global_noise)
            gen_samples = model.generate_det(z_samples)
            gen_samples = np.concatenate([imgs[i], gen_samples], axis=0)
            gen_samples = np.concatenate(gen_samples, axis=2)
            all_samples.append(gen_samples)
        all_samples = [np.concatenate(all_samples, axis=1)]
        utils.save_samples(samples_dir, all_samples, ['inv_gens'])
        all_samples = []
    for i in range(n_samples):
        global_noise = np.random.uniform(-1., 1., (1, config.nz_global, 1, 1))
        z_samples = utils.sample_noise_tensor(config, 5, config.zx,
                                              global_noise=global_noise)
        gen_samples = model.generate_det(z_samples)
        gen_samples = np.concatenate(gen_samples, axis=2)
        all_samples.append(gen_samples)
    all_samples = [np.concatenate(all_samples, axis=1)]
    utils.save_samples(samples_dir, all_samples, ['gens'])