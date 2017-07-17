import string
import skimage.io
import skimage.color
from skimage.transform import rescale
import os.path
from model import Alex
from image import clip_coin, show_bgrimg
import chainer.links as L
from chainer import serializers
from chainer import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cv2


def inference(img, model):
    IMAGE_SIZE = 128
    IN_CHANNELS = 3

    if 2 < len(img.shape) and img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img)
    height, width = img.shape[:2]
    img = rescale(img, (IMAGE_SIZE / height,
                        IMAGE_SIZE / width), mode='constant')
    im = img.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS)
    im = im.transpose(0, 3, 1, 2)
    x = Variable(im)
    y = model.predictor(x)
    [pred] = y.data
    print(pred)
    recog = np.argmax(pred)
    return recog, im.reshape(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).transpose(1, 2, 0)


def main():
    model = L.Classifier(Alex())
    serializers.load_npz('./model/model.npz', model)

    for r, ds, fs in os.walk('./sample/00/'):
        for f in fs:
            filename = os.path.join(r, f)
            image, imgs = clip_coin(filename)
            show_bgrimg(image)

            count = 1
            for img in imgs:
                plt.subplot(3, 4, count)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                recog, img = inference(img, model)
                plt.imshow(img)
                plt.title([
                    '  1_omote', '  1_ura',
                    '  5_omote', '  5_ura',
                    ' 10_omote', ' 10_ura',
                    ' 50_omote', ' 50_ura',
                    '100_omote', '100_ura',
                    '500_omote', '500_ura'][recog])
                count += 1
            plt.show()


if __name__ == '__main__':
    main()
