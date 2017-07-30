import os.path
from model import Alex
from image import clip_coin, adjust_gamma, normalize_image, show_bgrimg, IMAGE_SIZE
import chainer.links as L
from chainer import serializers
from chainer import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cv2


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def inference(img, model, in_channels):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if in_channels == 1:
        img = normalize_image(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = adjust_gamma(img)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255
    im = img.astype(np.float32).reshape(
        1, IMAGE_SIZE[0], IMAGE_SIZE[1], in_channels).transpose(0, 3, 1, 2)
    x = Variable(im)
    y = model.predictor(x)
    [pred] = y.data
    pred = softmax(pred)
    print(pred)
    return pred, img


def main():
    model = L.Classifier(Alex())
    serializers.load_npz('./model/model.npz', model)
    IN_CHANNELS = 3

    for r, ds, fs in os.walk('./sample/0001/'):
        for f in fs:
            filename = os.path.join(r, f)
            image, imgs = clip_coin(filename)
            show_bgrimg(image)

            count = 1
            for img in imgs:
                if 12 < count:
                    break
                plt.subplot(3, 4, count)
                pred, img_ = inference(img, model, IN_CHANNELS)
                recog = np.argmax(pred)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title([
                    '  1_omote', '  1_ura',
                    '  5_omote', '  5_ura',
                    ' 10_omote', ' 10_ura',
                    ' 50_omote', ' 50_ura',
                    '100_omote', '100_ura',
                    '500_omote', '500_ura'][recog])
                plt.axis('off')
                count += 1
            plt.show()


if __name__ == '__main__':
    main()
