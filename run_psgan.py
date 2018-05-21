import numpy as np
from tqdm import tqdm

from psgan import PSGAN
from data_io import save_tensor


def sample_noise_tensor(config, batch_size, zx, zx_quilt=None):
    Z = np.zeros((batch_size, config.nz, zx, zx))
    Z[:,
    config.nz_global:config.nz_global + config.nz_local] = np.random.uniform(
        -1., 1., (batch_size, config.nz_local, zx, zx))

    if zx_quilt is None:
        Z[:, :config.nz_global] = np.random.uniform(-1., 1., (
        batch_size, config.nz_global, 1, 1))
    else:
        for i in range(zx / zx_quilt):
            for j in range(zx / zx_quilt):
                Z[:, :config.nz_global, i * zx_quilt:(i + 1) * zx_quilt,
                j * zx_quilt:(j + 1) * zx_quilt] = np.random.uniform(-1., 1., (
                batch_size, config.nz_global, 1, 1))

    if config.nz_periodic > 0:
        for i, pixel in zip(range(1, config.nz_periodic + 1),
                            np.linspace(30, 130, config.nz_periodic)):
            band = np.pi * (0.5 * i / float(
                config.nz_periodic) + 0.5)  ##initial values for numerical stability
            ##just horizontal and vertical coordinate indices
            for h in range(zx):
                Z[:, -i * 2, :, h] = h * band
            for w in range(zx):
                Z[:, -i * 2 + 1, w] = w * band
    return Z


if __name__ == "__main__":
    psgan = PSGAN()
    c = psgan.config
    c.print_info()
    z_sample = sample_noise_tensor(c, 1, c.zx_sample, c.zx_sample_quilt)
    epoch = 0
    tot_iter = 0

    while epoch < c.epoch_count:
        epoch += 1
        print("Epoch %d" % epoch)

        Gcost = []
        Dcost = []

        iters = c.epoch_iters / c.batch_size
        iters = 1
        for it, samples in enumerate(tqdm(c.data_iter(), total=iters)):
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

        print "Gcost=", np.mean(Gcost), "  Dcost=", np.mean(Dcost)

        slist = []
        for img in samples:
            slist += [img]
        img = np.concatenate(slist, axis=2)
        save_tensor(img, 'samples/minibatchTrue_%s_epoch%d.jpg' % (
        c.save_name, epoch))

        samples = psgan.generate(Znp)
        slist = []
        for img in samples:
            slist += [img]
        img = np.concatenate(slist, axis=2)
        save_tensor(img, 'samples/minibatchGen_%s_epoch%d.jpg' % (
        c.save_name, epoch))

        data = psgan.generate(z_sample)

        save_tensor(data[0],
                    'samples/largesample%s_epoch%d.jpg' % (c.save_name, epoch))
        psgan.save('models/%s_epoch%d.psgan' % (c.save_name, epoch))
