import logging
from optparse import OptionParser
import os
import sys

from psgan import PSGAN
import utils
from train_and_sample import train


def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--mode", type='string', default='train',
                      help="train or sample")
    parser.add_option("--checkpoint", type='string', default=None,
                      help="load a model from checkpoint, format: \'Y-m-d.id\'")
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

    model_dir = utils.create_model_folder('models', vars(options))
    samples_dir = os.path.join(os.path.dirname(log_file), 'samples')

    checkpoint_path = utils.find_checkpoint('models', options.checkpoint)
    print checkpoint_path
    # psgan = PSGAN(checkpoint_path)

    # train(psgan, psgan.config, logger, options, model_dir, samples_dir)


if __name__ == '__main__':
    main()