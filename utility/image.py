import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2


IMAGE_SIZE = (160, 160)


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
        img = np.copy(img)
        img = (img - np.mean(img)) / np.std(img) * 64 + 128
    return img.astype(np.uint8)


def adjust_gamma(img):
    h, w, d = img.shape
    ave = img[int(0.2 * h):int(0.8 * h), int(0.2 * h):int(0.8 * h), :].mean()
    base = 5
    if ave < 128:
        gamma = -(base - 1) / 128 * ave + base
    else:
        gamma = (1 / base - 1) / 128 * ave + (2 - 1 / base)

    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    img = cv2.LUT(img, lookUpTable)
    return img


def square_check(contour):
    ''' 正方形っぽいときにTrueを戻す． '''
    x, y, w, h = contour
    return w * 0.9 < h and h < w * 1.1


def remove_illegal_coin(contours):
    ''' 面積が大きく他とはずれているものを省く '''
    if len(contours) < 3:
        return contours
    result = contours[:]
    areas = [w * h for x, y, w, h in contours]
    ave = np.array(areas).mean()
    remove_idxs = []
    for idx, area in enumerate(areas):
        others = [x for x in areas if x != area]
        if len(others) < 2:
            continue
        a = np.array(others).mean()
        if a < ave * 0.95 or ave * 1.05 < a:
            remove_idxs.append(idx)
    remove_idxs = remove_idxs[::-1]
    for i in remove_idxs:
        result.pop(i)
    return result


def remove_near_point(contours, img):
    ''' 座標が同じバウンディングボックスを省く '''
    height = img.shape[0]
    width = img.shape[1]
    result = []
    for contour in contours:
        x, y, w, h = contour
        for r in result:
            rx, ry, rw, rh = r
            if ((x - rx) ** 2 < (width * 0.05) ** 2) and ((y - ry) ** 2 < (height * 0.05) ** 2):
                break
        else:
            result.append(contour)
    return result


def remove_inner_box(contours):
    ''' 他のバウンディングボックスの内部に含まれるものを省く '''
    result = []
    for contour in contours:
        x, y, w, h = contour
        for other in contours:
            ox, oy, ow, oh = other
            if ox < x and x + w < ox + ow and oy < y and y + h < oy + oh:
                break
        else:
            result.append(contour)
    return result


def edge_image(img):
    ''' エッジ検出 '''
    img_sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    img_sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    img_abs_sobelx = cv2.convertScaleAbs(img_sobelx)
    img_abs_sobely = cv2.convertScaleAbs(img_sobely)
    img_sobel_edge = cv2.addWeighted(
        img_abs_sobelx, 0.5, img_abs_sobely, 0.5, 0)
    return img_sobel_edge


def clip_coin_partial(img, offset):
    '''
    渡された画像を切り取る．x，y座標は渡されたoffsetで補正する．
    '''
    one_chan_imgs = [cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), img[:, :, 0], img[:, :, 1], img[:, :, 2]]

    contours = []   # (x, y, w, h)のリスト
    for one_img in one_chan_imgs:
        ret, bin_img = cv2.threshold(
            one_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, contour, __ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += [cv2.boundingRect(x) for x in contour]

        bin_img = cv2.bitwise_not(bin_img)

        _, contour, __ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += [cv2.boundingRect(x) for x in contour]

    min_coin_area = img.shape[0] * img.shape[1] // 50
    contours = [x for x in contours if min_coin_area <
                x[2] * x[3] and square_check(x)]
    contours = remove_illegal_coin(contours)
    return [(x + offset[0], y + offset[1], w, h) for x, y, w, h in contours]


def clip_coin(filename):
    '''
    バウンディングボックスを描画した画像と，その部分を切り取った画像リストをtupleで戻す．
    '''
    img = cv2.imread(filename)

    contours = []   # (x, y, w, h)のリスト
    contours += clip_coin_partial(img, (0, 0))
    height, width, _ = img.shape
    contours += clip_coin_partial(img[:height // 2, :, :], (0, 0))
    contours += clip_coin_partial(img[height // 2:, :, :], (0, height // 2))
    contours += clip_coin_partial(img[:, :width // 2, :], (0, 0))
    contours += clip_coin_partial(img[:, width // 2:, :], (width // 2, 0))

    contours += clip_coin_partial(img[height // 4:height // 4 * 3, :, :], (0, height // 4))
    contours += clip_coin_partial(img[:, width // 4:width // 4 * 3, :], (width // 4, 0))

    contours += clip_coin_partial(img[:height // 2, :width // 2, :], (0, 0))
    contours += clip_coin_partial(img[:height // 2, width // 2:, :], (width // 2, 0))
    contours += clip_coin_partial(img[height // 2:, :width // 2, :], (0, height // 2))
    contours += clip_coin_partial(img[height // 2:, width // 2:, :], (width // 2, height // 2))

    contours += clip_coin_partial(img[height // 4:height // 4 * 3, :width // 2, :], (0, height // 4))
    contours += clip_coin_partial(img[height // 4:height // 4 * 3, width // 2:, :], (width // 2, height // 4))
    contours += clip_coin_partial(img[:height // 2, width // 4:width // 4 * 3, :], (width // 4, 0))
    contours += clip_coin_partial(img[height // 2:, width // 4:width // 4 * 3, :], (width // 4, height // 2))

    contours = remove_illegal_coin(contours)
    contours = remove_near_point(contours, img)

    clip_imgs = []
    boundingbox_img = np.copy(img)
    for contour in contours:
        x, y, w, h = contour
        n = min(w, h)
        clip_imgs.append(img[y:y + n, x:x + n])
        cv2.rectangle(boundingbox_img, (x, y), (x + n, y + n), (0, 255, 0), 10)

    return boundingbox_img, clip_imgs


def clip_all(root):
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
                resize_img = cv2.resize(img, IMAGE_SIZE)
                path, fname = os.path.split(filename)
                body, exe = os.path.splitext(fname)
                savename = os.path.join(
                    outdir, '{}_{:04d}{}'.format(body, count, '.png'))
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
    root = '../image/org/'
    for d in os.listdir(root):
        dirname = os.path.join(root, d)
        if os.path.isdir(dirname):
            rename(dirname)
            clip_all(dirname)


def main():
    create_sample_img()


if __name__ == '__main__':
    main()
