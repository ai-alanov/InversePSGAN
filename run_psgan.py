import numpy as np
from tqdm import tqdm
import logging
from optparse import OptionParser
import os
import sys

from psgan import PSGAN
import utils


def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--mode", type='string', default='train',
                      help="train or sample")
    parser.add_option("--checkpoint", type='string', default='last',
                      help="load a model from checkpoint")
    parser.add_option("--data", type='string', default='texture',
                      help="path to data for training")
    parser.add_option("--n_epochs", type='int', default=10,
                      help="how many epochs to do globally")
    parser.add_option("--n_iters", type='int', default=100,
                      help="steps inside one epoch")
    parser.add_option("--b_size", type='int', default=25,
                      help="batch size")
    (options, args) = parser.parse_args()

    log_file = utils.create_logging_file('logs', vars(options))
    logger = utils.create_logger('run_psgan', file=log_file, need_fmt=True)
    logger.setLevel(logging.INFO)

    sys.stderr = utils.copy_stream_to_log(sys.stderr, 'STDERR', log_file)

    psgan = PSGAN()
    c = psgan.config
    c.print_info()
    z_sample = utils.sample_noise_tensor(c, 1, c.zx_sample, c.zx_sample_quilt)

    samples_folder = os.path.join(os.path.dirname(log_file), 'samples')
    utils.makedirs(samples_folder)
    model_folder = utils.create_model_folder('models', vars(options))

    for epoch in tqdm(range(options.n_epochs), file=sys.stdout):
        logger.info("Epoch {}".format(epoch))
        Gcost = []
        Dcost = []

        for it in range(options.n_iters):
            Znp = utils.sample_noise_tensor(c, options.b_size, c.zx)

            if it % (c.k + 1) == 0:
                Gcost.append(psgan.train_g(Znp))
            else:
                samples = next(c.data_iter(options.b_size))
                Dcost.append(psgan.train_d(samples, Znp))
        msg = "Gcost = {}, Dcost = {}"
        logger.info(msg.format(np.mean(Gcost), np.mean(Dcost)))

        gen_samples = psgan.generate(Znp)
        large_sample = psgan.generate(z_sample)[0]
        utils.save_samples(samples_folder, epoch, samples,
                           gen_samples, large_sample)

        model_file = '{}_epoch{}.psgan'.format(c.save_name, epoch)
        psgan.save(os.path.join(model_folder, model_file))


if __name__ == '__main__':
    main()