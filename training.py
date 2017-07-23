from model import Alex
from image import adjust_gamma
import chainer
import chainer.function as F
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.datasets import tuple_dataset
from chainer.training import extensions
import os
import os.path
import numpy as np
import cv2


def load_data(dirname):
    IMAGE_SIZE = 128
    IN_CHANNELS = 3

    dirs = ['001_00', '001_01',
            '005_00', '005_01',
            '010_00', '010_01',
            '050_00', '050_01',
            '100_00', '100_01',
            '500_00', '500_01']

    count = 0
    for i, dir in enumerate(dirs):
        for r, ds, fs in os.walk(os.path.join(dirname, dir)):
            count += len(fs)
    xs = np.zeros((count, IN_CHANNELS, IMAGE_SIZE,
                   IMAGE_SIZE)).astype(np.float32)
    ys = np.zeros(count).astype(np.int32)

    idx = 0
    for i, dir in enumerate(dirs):
        for r, ds, fs in os.walk(os.path.join(dirname, dir)):
            for f in fs:
                filename = os.path.join(r, f)
                img = cv2.imread(filename)
                img = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB if IN_CHANNELS == 3 else cv2.COLOR_BGR2GRAY)
                img = adjust_gamma(img)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = img / 255
                im = img.astype(np.float32).reshape(
                    IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS).transpose(2, 0, 1)
                xs[idx] = im
                ys[idx] = i
                idx += 1

    return tuple_dataset.TupleDataset(xs, ys)


def main():
    model = L.Classifier(Alex())
    modelfile = './model/model.npz'
    if os.path.isfile(modelfile):
        serializers.load_npz(modelfile, model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optfile = './model/optimizer.npz'
    if os.path.isfile(optfile):
        serializers.load_npz(optfile, optimizer)

    train = load_data('./image/train')
    test = load_data('./image/test')
    train_iter = chainer.iterators.SerialIterator(train, batch_size=100)
    test_iter = chainer.iterators.SerialIterator(
        test, batch_size=100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=None)
    trainer = training.Trainer(updater, (60, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=None))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    modeldir = './model'
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    serializers.save_npz(os.path.join(modeldir, 'model.npz'), model)
    serializers.save_npz(os.path.join(modeldir, 'optimizer.npz'), optimizer)


if __name__ == '__main__':
    main()
