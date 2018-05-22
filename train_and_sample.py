import numpy as np
import os
import sys
from tqdm import tqdm

import utils


def train(model, config, logger, options, model_dir, samples_dir):
    z_sample = utils.sample_noise_tensor(config, 1, config.zx_sample,
                                         config.zx_sample_quilt)

    utils.makedirs(samples_dir)

    for epoch in tqdm(range(options.n_epochs), file=sys.stdout):
        logger.info("Epoch {}".format(epoch))
        Gcost = []
        Dcost = []

        for it in range(options.n_iters):
            Znp = utils.sample_noise_tensor(config, options.b_size, config.zx)

            if it % (config.k + 1) == 0:
                Gcost.append(model.train_g(Znp))
            else:
                samples = next(config.data_iter(options.b_size))
                Dcost.append(model.train_d(samples, Znp))
        msg = "Gcost = {}, Dcost = {}"
        logger.info(msg.format(np.mean(Gcost), np.mean(Dcost)))

        Znp = utils.sample_noise_tensor(config, options.b_size, config.zx)
        samples = next(config.data_iter(options.b_size))

        gen_samples = model.generate(Znp)
        large_sample = model.generate(z_sample)[0]
        utils.save_samples(samples_dir, epoch, samples,
                           gen_samples, large_sample)

        model_file = 'epoch_{}.model'.format(epoch)
        model.save(os.path.join(model_dir, model_file))