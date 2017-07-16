import os
import time
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_SIZE = 64


def show_bgrimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def show_grayimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()


def clip_coin(filename):
    '''
    背景を黒にしてコイン部分を枠で囲う．
    バウンディングボックスを描画した画像と，その部分を切り取った画像リストをtupleで戻す．
    '''

    img = cv2.imread(filename)
    h, w, d = img.shape
    img = img[3:h - 3, 3:w - 3]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    is_white_base = 128 < gaus_img[10, 10]
    _, bin_img = cv2.threshold(
        gaus_img, 222 if is_white_base else 60, 255, cv2.THRESH_BINARY)

    if is_white_base:
        bin_img = cv2.bitwise_not(bin_img)    # 背景が白の場合は必要

    _, coins_contours, __ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_img = np.zeros(img.shape, dtype=np.uint8)  # 黒背景のベース
    min_coin_area = 60
    large_contours = [
        x for x in coins_contours if min_coin_area < cv2.contourArea(x)]
    for x in large_contours:
        cv2.fillConvexPoly(mask_img, x, (255, 255, 255))    # 黒背景のベースにコイン部を白で描画

    masked_img = cv2.bitwise_and(img, mask_img)     # mask_imgの白部分のみimgを描画

    clip_imgs = []
    for contour in large_contours:
        x, y, w, h = cv2.boundingRect(contour)
        n = min(w, h)

        if n < MIN_SIZE:
            continue

        clip_img = np.copy(masked_img[y:y + n, x:x + n])
        clip_imgs.append(clip_img)
        cv2.rectangle(masked_img, (x, y), (x + n, y + n), (0, 255, 0), 2)

    return masked_img, clip_imgs


def clip_all(root):
    outdir = os.path.join(root, 'out')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if os.path.isfile(filename):
            _, imgs = clip_coin(filename)
            for img in imgs:
                path, fname = os.path.split(filename)
                body, exe = os.path.splitext(fname)
                savename = os.path.join(
                    outdir, '{}_{}{}'.format(body, id(img), '.png'))
                cv2.imwrite(savename, img)


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
