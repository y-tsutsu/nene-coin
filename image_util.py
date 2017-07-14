import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = './image/origin/100_01/data/'


def rename():
    count = 0
    for r, ds, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(r, f)
            body, exe = os.path.splitext(f)
            newname = os.path.join(r, '{0:04d}{1}'.format(count, exe))
            os.rename(filename, newname)
            count += 1


def convert_png():
    count = 0
    for r, ds, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            newname = os.path.join(r, '{0:04d}.png'.format(count))
            cv2.imwrite(newname, img)
            count += 1


def clip_img():
    for r, ds, fs in os.walk(path):
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


def resize_img():
    for r, ds, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            height, width, channels = img.shape
            img = cv2.resize(img, (100, 100))
            cv2.imwrite(filename, img)


def rotate_img():
    for r, ds, fs in os.walk('image/origin/100_01'):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            size = tuple([img.shape[1], img.shape[0]])
            center = tuple([int(size[0] / 2), int(size[1] / 2)])
            scale = 1.0
            for x in range(0, 36):
                angle = float(x * 10)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
                r_img = cv2.warpAffine(
                    img, rotation_matrix, size, flags=cv2.INTER_CUBIC)
                f1, f2 = os.path.splitext(filename)
                savename = '{0}/data/{1}_{2:03d}{3}'.format(r, os.path.split(f1)[1], int(angle), f2)
                cv2.imwrite(savename, r_img)


def preprocessing():
    for r, ds, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gaus_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
            _, bin_img = cv2.threshold(gaus_img, 170, 255, cv2.THRESH_BINARY)
            bin_img = cv2.bitwise_not(bin_img)    # 背景が白の場合は必要

            _, coins_contours, __ = cv2.findContours(
                bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_img = np.zeros(img.shape, dtype=np.uint8)          # 黒背景のベース
            min_coin_area = 60
            large_contours = [
                cnt for cnt in coins_contours if cv2.contourArea(cnt) > min_coin_area]
            cv2.fillConvexPoly(mask_img,
                               large_contours[0], (255, 255, 255))  # コインを白で描画

            show_img = cv2.bitwise_and(img, mask_img)
            show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
            plt.imshow(show_img)
            plt.show()


def split_train_test():
    os.mkdir(os.path.join(path, 'test'))
    os.mkdir(os.path.join(path, 'train'))
    count = 0
    for r, ds, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(r, f)
            if count % 6 == 1:
                shutil.move(filename, os.path.join(r, 'test', f))
            else:
                shutil.move(filename, os.path.join(r, 'train', f))
            count += 1


def main():
    split_train_test()


if __name__ == '__main__':
    main()
