import os
import time
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_bgrimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def show_grayimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()


def draw_hole_black(img):
    '''
    5円，50円の穴を黒でぬる．
    '''
    img = np.copy(img)
    w_min = np.array([150, 150, 150], np.uint8)
    w_max = np.array([255, 255, 255], np.uint8)
    bin_img = cv2.inRange(img, w_min, w_max)
    # show_grayimg(hoge_img)

    _, coins_contours, __ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_coin_area = 60

    def inner_check(img, contour):
        '''
        contourがimgの中央あたりにあるかチェックする．（5円，50円のくり抜き用）
        '''
        def inner(a, b):
            return a * 0.8 < b and b < a * 1.2
        height, width, depth = img.shape
        x, y, w, h = cv2.boundingRect(contour)
        return inner(x, y) and inner(x, width - (x + w)) and inner(y, height - (y + h)) and w < width * 0.35

    large_contours = [x for x in coins_contours if min_coin_area <
                      cv2.contourArea(x) and inner_check(img, x)]

    for x in large_contours:
        cv2.fillConvexPoly(img, x, (0, 0, 0))  # 黒背景のベースにコイン部を白で描画

    return img


def clip_coin(filename):
    '''
    バウンディングボックスを描画した画像と，その部分を切り取った画像リストをtupleで戻す．
    '''
    img = cv2.imread(filename)
    h, w, d = img.shape

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    is_white_base = 128 < gaus_img[10, 10]
    _, bin_img = cv2.threshold(
        gaus_img, 192 if is_white_base else 60, 255, cv2.THRESH_BINARY)

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

        clip_img = np.copy(masked_img[y:y + n, x:x + n])
        clip_img = draw_hole_black(clip_img)
        clip_imgs.append(clip_img)
        cv2.rectangle(img, (x, y), (x + n, y + n), (0, 255, 0), 2)

    return img, clip_imgs


def clip_all(root):
    IMAGE_SIZE = 128

    outdir = os.path.join(root, 'out')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if os.path.isfile(filename):
            _, imgs = clip_coin(filename)
            for img in imgs:
                resize_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                path, fname = os.path.split(filename)
                body, exe = os.path.splitext(fname)
                savename = os.path.join(
                    outdir, '{}_{}{}'.format(body, id(img), '.png'))

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
