import os
import time
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_rgbimg(img):
    plt.imshow(img)
    plt.show()


def show_bgrimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def show_grayimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()


def adjust_gamma(img):
    ave = img.mean()
    base = 3
    if ave < 128:
        gamma = -(base - 1) / 128 * ave + base
    else:
        gamma = (1 / base - 1) / 128 * ave + (base - 1 - 1 / base)

    print(ave, gamma)

    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    return cv2.LUT(img, lookUpTable)


def clip_coin(filename):
    '''
    バウンディングボックスを描画した画像と，その部分を切り取った画像リストをtupleで戻す．
    '''
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if 255 / 2 < bin_img.mean():
        bin_img = cv2.bitwise_not(bin_img)

    _, coins_contours, __ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_coin_area = 10000
    large_contours = [
        x for x in coins_contours if min_coin_area < cv2.contourArea(x)]

    def inner_check(a, b):
        return a * 0.8 < b and b < a * 1.2

    clip_imgs = []
    boundingbox_img = np.copy(img)
    for contour in large_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not inner_check(w, h):
            continue
        n = min(w, h)
        clip_imgs.append(img[y:y + n, x:x + n])
        cv2.rectangle(boundingbox_img, (x, y), (x + n, y + n), (0, 255, 0), 10)

    return boundingbox_img, clip_imgs


def clip_all(root):
    IMAGE_SIZE = 128

    outdir = os.path.join(root, 'out')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    count = 0
    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if os.path.isfile(filename):
            _, imgs = clip_coin(filename)
            for img in imgs:
                resize_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                path, fname = os.path.split(filename)
                body, exe = os.path.splitext(fname)
                savename = os.path.join(
                    outdir, '{}_{}{}'.format(body, count, '.png'))
                count += 1
                cv2.imwrite(savename, resize_img)


def rename(root):
    count = 0
    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if os.path.isfile(filename):
            body, exe = os.path.splitext(f)
            newname = os.path.join(
                root, '{0:04d}{1}'.format(count, exe.lower()))
            os.rename(filename, newname)
            count += 1


def create_sample_img():
    root = './image_next/'
    for d in os.listdir(root):
        dirname = os.path.join(root, d)
        if os.path.isdir(dirname):
            rename(dirname)
            clip_all(dirname)


def main():
    create_sample_img()


if __name__ == '__main__':
    main()
