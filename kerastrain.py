from image import adjust_gamma, normalize_image, IMAGE_SIZE
import os
import os.path
import numpy as np
import cv2
from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical


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
                # im = img.astype(np.float32).reshape(
                #     IMAGE_SIZE[0], IMAGE_SIZE[1], in_channels).transpose(2, 0, 1)
                xs[idx] = img
                label = np.zeros(len(dirs)).astype(np.int32)
                label[i] = 1
                ys[idx] = label
                idx += 1

    return xs, ys


def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )


def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )


def main():
    model = Sequential()

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4, 4),
                     bias_init=0, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(12, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    IN_CHANNELS = 3
    train = load_data('./image/train', IN_CHANNELS)
    test = load_data('./image/test', IN_CHANNELS)

    check = ModelCheckpoint('./model/model.hdf5')
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = model.fit(train[0], train[1], epochs=15, batch_size=100,
                        shuffle=True, validation_split=0.25, callbacks=[early_stopping])


if __name__ == '__main__':
    main()
