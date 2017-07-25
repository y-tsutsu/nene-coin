import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image import show_bgrimg, show_grayimg, IMAGE_SIZE
from increase_picture import increase_picture

path = './image/origin/100_01/'


def convert_png(dirname):
    count = 0
    for r, ds, fs in os.walk(dirname):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            newname = os.path.join(r, '{0:04d}.png'.format(count))
            cv2.imwrite(newname, img)
            count += 1


def clip_img(dirname):
    for r, ds, fs in os.walk(dirname):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            h, w, d = img.shape
            if h < w:
                x = (w - h) // 2
                img = img[:, x:x + h]
            else:
                x = (h - w) // 2
                img = img[x:x + w, :]
            cv2.imwrite(filename, img)


def resize_img(dirname):
    for r, ds, fs in os.walk(dirname):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            height, width, channels = img.shape
            img = cv2.resize(img, IMAGE_SIZE)
            cv2.imwrite(filename, img)


def rotate_img(dirname):
    for r, ds, fs in os.walk(dirname):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            size = tuple([img.shape[1], img.shape[0]])
            center = tuple([int(size[0] / 2), int(size[1] / 2)])
            scale = 1.0
            for x in range(0, 4):
                angle = float(x * 90)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
                r_img = cv2.warpAffine(img, rotation_matrix, size, dst=np.copy(img),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
                f1, f2 = os.path.splitext(filename)
                savename = '{0}/{1}_{2:03d}{3}'.format(
                    r, os.path.split(f1)[1], int(angle), f2)
                cv2.imwrite(savename, r_img)
            os.remove(filename)


def increase_img(dirname):
    for r, ds, fs in os.walk(dirname):
        for f in fs:
            filename = os.path.join(r, f)
            img_src = cv2.imread(filename, cv2.IMREAD_COLOR)
            imgs = increase_picture(img_src)
            for i, img in enumerate(imgs):
                f1, f2 = os.path.splitext(filename)
                savename = '{0}/{1}_{2:02d}{3}'.format(
                    r, os.path.split(f1)[1], i, f2)
                cv2.imwrite(savename, img)
            os.remove(filename)


def main():
    resize_img('./image/train/')
    resize_img('./image/test/')
    increase_img('./image/train/')
    increase_img('./image/test/')
    rotate_img('./image/train/')
    rotate_img('./image/test/')


if __name__ == '__main__':
    main()
