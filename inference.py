import string
import skimage.io
import skimage.color
from skimage.transform import rescale
import os.path
from model import CNN
import chainer.links as L
from chainer import serializers
from chainer import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn


IMAGE_SIZE = 128


def inference(img):
    model = L.Classifier(CNN())
    serializers.load_npz('./model/model.npz', model)

    if img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img)
    height, width = img.shape[:2]
    img = rescale(img, (IMAGE_SIZE / height,
                        IMAGE_SIZE / width), mode='constant')
    im = img.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    im = im.transpose(0, 3, 1, 2)
    x = Variable(im)
    y = model.predictor(x)
    [pred] = y.data
    print(pred)
    recog = np.argmax(pred)
    return recog, im.reshape(3, IMAGE_SIZE, IMAGE_SIZE).transpose(1, 2, 0)


def main():
    for r, ds, fs in os.walk('./sample'):
        for f in fs:
            filename = os.path.join(r, f)
            img = skimage.io.imread(filename)
            recog, img = inference(img)
            plt.title(['  1_0', '  1_1',
                       '  5_0', '  5_1',
                       ' 10_0', ' 10_1',
                       ' 50_0', ' 50_1',
                       '100_0', '100_1',
                       '500_0', '500_1'][recog], size=12)
            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    main()
