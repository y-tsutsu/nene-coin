import os
import os.path
import chainer.links as L
from chainer import serializers
from model import Alex
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from inference import inference
from image import clip_coin, show_bgrimg


def load_model():
    model = L.Classifier(Alex())
    serializers.load_npz('./model/model.npz', model)
    print('complete load model.')
    return model


def demo(model, dirname):
    root = os.path.join('./sample/', dirname)
    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if not os.path.isfile:
            continue
        if os.path.splitext(filename)[1] == '.db':
            continue
        image, imgs = clip_coin(filename)
        print(filename)
        show_bgrimg(image)
        for img in imgs:
            show_bgrimg(img)
            recog, img = inference(img, model)
            print([
                '!!!   1円 表 !!!', '!!!   1円　裏 !!!',
                '!!!   5円 表 !!!', '!!!   5円　裏 !!!',
                '!!!  10円 表 !!!', '!!!  10円　裏 !!!',
                '!!!  50円 表 !!!', '!!!  50円　裏 !!!',
                '!!! 100円 表 !!!', '!!! 100円　裏 !!!',
                '!!! 500円 表 !!!', '!!! 500円　裏 !!!'][recog])


def main():
    model = load_model()
    demo(model, '0001')


if __name__ == '__main__':
    main()
