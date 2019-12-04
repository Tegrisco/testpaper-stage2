import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import scipy
# from src.base import data_dir
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import sys
sys.path.append('../packages/svm_model_v0_2/')
import helpers as hp
import imutils as imu


def detect_vertical_line(img, gray, margin=0):
    """

    :param img:
    :param gray:
    :param margin: 纠偏时，上下左右各去除 margin，是一个比例 0<margin<1.0
    :return:
    """
    # Hough 变换取最长线作为基线来纠偏，本质上是不太精确的。
    # 另外因为试卷中可能有图形，最长线可能是图形中的线，角度随机。
    # 例：12-2

    h, w = img.shape[:2]

    sy = int(h * margin)
    ey = int(h * (1-margin))
    sx = int(w * margin)
    ex = int(w * (1-margin))

    gray = gray[sy:ey, sx:ex]
    # imu.imshow_(gray)

    # canny 的这两个参数非常重要， double 类型
    edges = cv2.Canny(gray, 500, 1000)
    # imu.imshow_(edges)

    # h, theta, d = hough_line(edges, theta=np.linspace(-np.pi / 2, np.pi / 2, 180 * 5, endpoint=False))
    h, theta, d = hough_line(edges, theta=np.linspace(-np.pi / 180, np.pi / 180, 11, endpoint=True))
    h, theta, d = hough_line_peaks(h, theta, d, num_peaks=20)

    for i in range(len(h)):
        a = np.cos(theta[i])
        b = np.sin(theta[i])
        x0, y0 = a * d[i], b * d[i]
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('./tmp/x.jpg', img)

    # 注意返回值是 lines，而不是 rho, theta
    # lines = cv2.HoughLinesP(edges, 1, np.pi / (180*5), 100, minLineLength=60, maxLineGap=2)
    #
    # length = 0
    # angle = 0
    # tol = 5.0/180.0*np.pi
    #
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     tmp = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     angle_tmp = np.arctan2(y1 - y2, x1 - x2)  #特别要注意 arctan2 的参数说明,xy轴取cartesian方式
    #     if angle_tmp < 0:
    #         angle_tmp = angle_tmp + np.pi
    #     if tmp > length and ( angle_tmp<(np.pi/2 + tol)  or angle_tmp>(np.pi/2-tol)   ):
    #         length = tmp
    #         longest_line = (x1, y1, x2, y2)
    #         angle = angle_tmp

    DEBUG = True
    if DEBUG:
        (x1, y1, x2, y2) = longest_line
        x1 = x1+sx
        x2 = x2+sx
        y1 = y1+sy
        y2 = y2+sy
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('./tmp/x.jpg', img)


    if angle > np.pi/2:
        angle = angle - np.pi

    if abs(angle) < 0.05 / 180 *np.pi:
        return img, 0.0

    angle = angle * 180/np.pi

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def detect_vertical_line2(gray):
    # grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binImg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # imu.imshow_(binImg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)

    # imu.imwrite_(binImg, 0)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 600))
    dest = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)

    # imu.imwrite_(dest, 1)


    element = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    dilation = cv2.dilate(dest, element, iterations=2)

    # imu.imwrite_(dilation, 2)

    return cv2.bitwise_or(gray, dilation)


def detect_vertical_line3(gray, thres=2000):
    # grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binImg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # imu.imshow_(binImg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    binImg = cv2.morphologyEx(binImg, cv2.MORPH_DILATE, kernel)

    binImg = binImg // 255
    weight = np.sum(binImg, axis=0)

    # plt.plot(weight)
    # plt.show()

    left = 0
    cleft = 0
    cright =0
    right = gray.shape[1]
    level1 = gray.shape[1]//3
    level2 = 2*level1

    idxes = np.nonzero(weight > thres)[0]

    group1 = idxes[np.nonzero(idxes<level1)[0]]
    if group1.size > 0:
        left = group1.max()

    group2 = idxes[np.nonzero(np.logical_and(idxes>level1, idxes<level2))[0]]
    if group2.size > 0:
        cleft = group2.min()
        cright = group2.max()


    group3 = idxes[np.nonzero(idxes>level2)[0]]
    if group3.size > 0:
        right = group3.min()


    gray[:,0:left]=255
    gray[:, cleft:cright]=255
    gray[:, right:]=255

    # print('good')
    # imu.imwrite_(binImg, 0)


    return gray


def test_detect_vertical_line():

    books = ['biaozhundashijuan', 'jiangsumijuan', 'liangdiangeili']
    out_base = hp.mkdir('../result/vertical')

    for book in books:

        srcdir = os.path.join(data_dir, 'waibu/' + book)
        outdir = hp.mkdir(os.path.join(out_base, book))

        papers = os.listdir(srcdir)
        # papers = ['120190703153847968.jpg']
        for paper in papers:
            if paper.endswith('.jpg'):
                print(paper)
                src = cv2.imread(os.path.join(srcdir, paper))
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                vline = detect_vertical_line3(gray)
                # rotated, angle = correct_skew3(src, gray)

                cv2.imwrite(os.path.join(outdir, paper), vline)









if __name__ == "__main__":
    test_detect_vertical_line()

