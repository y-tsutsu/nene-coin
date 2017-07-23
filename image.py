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


def normalize_image(img):
    if len(img.shape) == 3:
        img = np.copy(img)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        r = (r - np.mean(r)) / np.std(r) * 64 + 128
        g = (g - np.mean(g)) / np.std(g) * 64 + 128
        b = (b - np.mean(b)) / np.std(b) * 64 + 128
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
    else:
        img = (img - np.mean(img)) / np.std(img) * 64 + 128
    return img.astype(np.uint8)


def adjust_gamma(img):
    ave = img.mean()
    base = 3
    if ave < 128:
        gamma = -(base - 1) / 128 * ave + base
    else:
        gamma = (1 / base - 1) / 128 * ave + (base - 1 - 1 / base)

    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    img = cv2.LUT(img, lookUpTable)
    img = normalize_image(img)
    return img


def square_check(contour):
    ''' 正方形っぽいときにTrueを戻す． '''
    x, y, w, h = cv2.boundingRect(contour)
    return w * 0.8 < h and h < w * 1.2


def remove_near_point(contours, img):
    ''' 座標が同じバウンディングボックスを省く '''
    height = img.shape[0]
    width = img.shape[1]
    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for r in result:
            rx, ry, rw, rh = cv2.boundingRect(r)
            if ((x - rx) ** 2 < (width * 0.05) ** 2) and ((y - ry) ** 2 < (height * 0.05) ** 2):
                break
        else:
            result.append(contour)
    return result


def remove_inner_box(contours):
    ''' 他のバウンディングボックスの内部に含まれるものを省く '''
    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for other in contours:
            ox, oy, ow, oh = cv2.boundingRect(other)
            if ox < x and x < ox + ow and oy < y and y < oy + oh:
                break
        else:
            result.append(contour)
    return result


def clip_coin(filename):
    '''
    バウンディングボックスを描画した画像と，その部分を切り取った画像リストをtupleで戻す．
    '''
    img = cv2.imread(filename)
    one_chan_imgs = [cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), img[:, :, 0], img[:, :, 1], img[:, :, 2]]

    contours = []
    for one_img in one_chan_imgs:
        ret, bin_img = cv2.threshold(
            one_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if 255 / 2 < bin_img.mean():
            bin_img = cv2.bitwise_not(bin_img)

        _, x, __ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours += x

    min_coin_area = img.shape[0] * img.shape[1] // 1000
    contours = [x for x in contours if min_coin_area <
                cv2.contourArea(x) and square_check(x)]
    contours = remove_near_point(contours, img)
    contours = remove_inner_box(contours)

    clip_imgs = []
    boundingbox_img = np.copy(img)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
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
    root = './sample/'
    for d in os.listdir(root):
        dirname = os.path.join(root, d)
        if os.path.isdir(dirname):
            rename(dirname)
            clip_all(dirname)


def main():
    create_sample_img()


if __name__ == '__main__':
    main()
