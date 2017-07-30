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
    model_gray = L.Classifier(Alex())
    serializers.load_npz('./model/model_gray.npz', model_gray)
    model_color = L.Classifier(Alex())
    serializers.load_npz('./model/model_color.npz', model_color)
    print('Load completed!!')
    return model_gray, model_color


def demo(models, dirname):
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
            pred_gray, img_gray = inference(img, models[0], 1)
            pred_color, img_color = inference(img, models[1], 3)
            recog_gray = np.argmax(pred_gray)
            recog_color = np.argmax(pred_color)
            title = ['  1円 表', '  1円　裏', '  5円 表', '  5円　裏', ' 10円 表', ' 10円　裏',
                     ' 50円 表', ' 50円　裏', '100円 表', '100円　裏', '500円 表', '500円　裏']
            if recog_gray == recog_color:
                print('*** {} ***'.format(title[recog_gray]))
            elif pred_gray[recog_gray] < pred_color[recog_color]:
                print('### {} or ({}) ###'.format(
                    title[recog_color], title[recog_gray]))
            else:
                print('=== {} or ({}) ==='.format(
                    title[recog_gray], title[recog_color]))
        print()


def main():
    models = load_model()
    demo(models, '0001')


if __name__ == '__main__':
    main()
