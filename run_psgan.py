import logging
from optparse import OptionParser
import os
import sys

from psgan import PSGAN, InversePSGAN, InversePSGAN2, InversePSGAN3
import utils
from train_and_sample import train, sample


def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--mode", type='string', default='train',
                      help="train or sample")
    parser.add_option("--inverse", type='int', default=0,
                      help="use PSGAN or InversePSGAN")
    parser.add_option("--checkpoint", type='string', default=None,
                      help="load a model from checkpoint, format: \'Y-m-d.id\'")
    parser.add_option("--data", type='string',
                      help="path to data for training")
    parser.add_option("--n_epochs", type='int', default=60,
                      help="how many epochs to do globally")
    parser.add_option("--n_iters", type='int', default=1000,
                      help="steps inside one epoch")
    parser.add_option("--b_size", type='int', default=40,
                      help="batch size")
    parser.add_option("--z_rec_fac", type='float', default=0.0,
                      help="factor for Z reconstruction loss")
    parser.add_option("--x_rec_fac", type='float', default=0.0,
                      help="factor for X reconstruction loss")
    parser.add_option("--save_step", type='int', default=10,
                      help="step for saving model")
    parser.add_option("--k", type='int', default=1,
                      help="number of G updates vs D updates")
    parser.add_option("--nz_g", type='int', default=60,
                      help="dimension of z global")
    (options, args) = parser.parse_args()

    log_file = utils.create_logging_file('logs', vars(options))
    logger = utils.create_logger('run_psgan', file=log_file, need_fmt=True)
    logger.setLevel(logging.INFO)

    sys.stderr = utils.copy_stream_to_log(sys.stderr, 'STDERR', log_file)

    checkpoint_path = utils.find_checkpoint('models', options.checkpoint)
    if options.inverse == 0:
        psgan = PSGAN(checkpoint_path)
    elif options.inverse == 1:
        psgan = InversePSGAN(checkpoint_path, z_reconst_fac=options.z_rec_fac,
                             x_reconst_fac=options.x_rec_fac)
    elif options.inverse == 2:
        psgan = InversePSGAN2(checkpoint_path, k=options.k,
                              nz_global=options.nz_g)
    elif options.inverse == 3:
        psgan = InversePSGAN3(checkpoint_path)
    else:
        raise Exception('No valid inverse parameter!')

    if options.mode == 'train':
        model_dir = utils.create_model_folder('models', vars(options))
        samples_dir = os.path.join(os.path.dirname(log_file), 'samples')

        train(psgan, psgan.config, logger, options, model_dir,
              samples_dir, inverse=options.inverse, save_step=options.save_step)
    elif options.mode == 'sample':
        samples_dir = os.path.join(os.path.dirname(log_file), 'samples')
        sample(psgan, psgan.config, samples_dir,
               options.data, inverse=options.inverse)


if __name__ == '__main__':
    main()
