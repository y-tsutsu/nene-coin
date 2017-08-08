import os
import sys
from model import get_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../utility/'))
from image import adjust_gamma, normalize_image, IMAGE_SIZE


def load_data(dirname, in_channels):
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
    xs = np.zeros((count, IMAGE_SIZE[0], IMAGE_SIZE[1], in_channels)).astype(
        np.float32)
    ys = np.zeros((count, len(dirs))).astype(np.int32)

    idx = 0
    for i, dir in enumerate(dirs):
        for r, ds, fs in os.walk(os.path.join(dirname, dir)):
            for f in fs:
                filename = os.path.join(r, f)
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if in_channels == 1:
                    img = normalize_image(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    img = adjust_gamma(img)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255
                im = img.astype(np.float32).reshape(
                    IMAGE_SIZE[0], IMAGE_SIZE[1], in_channels)
                xs[idx] = im
                label = np.zeros(len(dirs)).astype(np.int32)
                label[i] = 1
                ys[idx] = label
                idx += 1

    return xs, ys


def main():
    IN_CHANNELS = 3
    model = get_model(IN_CHANNELS)

    modeldir = './model'
    modelfile = os.path.join(modeldir, 'model.hdf5')
    if os.path.isfile(modelfile):
        model.load_weights(modelfile)

    train = load_data('../image/train', IN_CHANNELS)
    test = load_data('../image/test', IN_CHANNELS)

    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    check = ModelCheckpoint(modelfile)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = model.fit(train[0], train[1], epochs=30, batch_size=100, shuffle=True,
                        verbose=1, validation_data=(test[0], test[1]), callbacks=[check, early_stopping])


if __name__ == '__main__':
    main()
