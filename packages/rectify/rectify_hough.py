import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import scipy
# from src.base import data_dir
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import sys
sys.path.append('../packages/rectify/')
import imutils as imu

def correct_skew2(img, gray, margin=0):
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
    # accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=1)

    # 注意返回值是 lines，而不是 rho, theta
    lines = cv2.HoughLinesP(edges, 1, np.pi / (180*5), 100, minLineLength=60, maxLineGap=2)
    length = 0
    angle = 0
    tol = 5.0/180.0*np.pi

    for line in lines:
        x1, y1, x2, y2 = line[0]
        tmp = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        angle_tmp = np.arctan2(y1 - y2, x1 - x2)  #特别要注意 arctan2 的参数说明,xy轴取cartesian方式
        if angle_tmp < 0:
            angle_tmp = angle_tmp + np.pi
        if tmp > length and ( angle_tmp<tol  or angle_tmp>(np.pi-tol)   ):
            length = tmp
            longest_line = (x1, y1, x2, y2)
            angle = angle_tmp

    DEBUG = False
    if DEBUG:
        (x1, y1, x2, y2) = longest_line
        x1 = x1+sx
        x2 = x2+sx
        y1 = y1+sy
        y2 = y2+sy
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


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


def correct_skew3(img, gray):
    #放弃，角度不精确

    h, w = img.shape[:2]

    sy = int(h * 0.1)
    ey = int(h * 0.9)
    sx = int(w * 0.1)
    ex = int(w * 0.9)

    gray = gray[sy:ey, sx:ex]
    # imu.imshow_(gray)

    edges = cv2.Canny(gray, 100, 200)
    # cv2.imwrite("tmp/edge.jpg",edges)

    # gray = gray[523:523+175, 2566:2566+710]
    # edges = edges[523:523+175, 2566:2566+710]
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)


    # cv2.imwrite("tmp/edge.jpg",edges)

    angles = np.arctan2(sobely, sobelx)
    angles = angles/np.pi*180


    idx = angles < 0
    angles[idx] = angles[idx] + 180
    idx = edges == 0
    angles[idx] = -1

    # angles = angles.astype(int)
    # np.savetxt('tmp/edge.txt', angles, fmt='%4d')
    # cv2.imwrite('tmp/angle.jpg',angles/180.0)

    num_bins = 100 + 1
    lowbound = 85
    upbound = 95
    centers = np.linspace(lowbound, upbound, num_bins)
    span  = (upbound - lowbound)/(num_bins-1)
    lowbound = lowbound - span/2
    upbound = upbound + span/2


    hist, bin_edges = np.histogram(angles, num_bins, (lowbound,upbound))
    idx  = np.argmax(hist)
    print(centers[idx])
    print(hist)
    # hist = scipy.signal.medfilt(hist,3)

    angle = centers[idx]
    angle = angle - 90
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def test_correct_skew(srcdir,outdir):
    '''
    #srcdir = 'F:/exam_dataset/waibu/waibu-liangdiangeili'
    srcdir='../result/lean_biao'

    # outdir = '../result/rotate'
    outdir = '../result/lean_biao_hough'
'''
    papers = os.listdir(srcdir)
    #papers = ['120190703153513729.jpg']

    for paper in papers:
        imagesavePath = os.path.join(outdir, paper)
        if paper.endswith('.jpg'):
            print(paper)
            src = cv2.imread(os.path.join(srcdir, paper))
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            rotated, angle = correct_skew2(src, gray)
            # rotated, angle = correct_skew3(src, gray)
            print(angle)
            cv2.imwrite(os.path.join(outdir, paper), rotated)
    return  outdir



def test_crop_vertical_stipe():
    # srcdir = r'd:\autograde\dataset\tz\tz-grade1-2017-1-math'
    # papers = os.listdir(srcdir)
    # papers = ['12019062619022377220190626195127221.jpg']

    srcdir = 'F:/exam_dataset/waibu/waibu-biaozhundashijuan'
    papers = os.listdir(srcdir)
    #papers = ['120190703153513729.jpg']

    outdir = '../result/waibu-biaozhundashijuan_rotate'


    for paper in papers:
        if paper.endswith('.jpg'):
            print(paper)
            src = cv2.imread(os.path.join(srcdir, paper))
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            rotated, angle = correct_skew2(src, gray, 0.1)
            # rotated, angle = correct_skew3(src, gray)

            cv2.imwrite(os.path.join(outdir, paper), rotated)

def generate():
    # srcdir = '../img/rotate'
   # srcdir = '../img/paper/empty'
    #srcdir =os.path.join(data_dir, 'waibu\waibu-biaozhundashjuan')
    srcdir='F:/exam_dataset/waibu/waibu-biaozhundashijuan'
    outdir = '../result/lean_biao'
    papers = os.listdir(srcdir)
    for paper in papers:
        if paper.endswith('.jpg'):
            print(paper)
            img = cv2.imread(os.path.join(srcdir, paper))
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)

            angle = 8*np.random.random() -4

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            cv2.imwrite(os.path.join(outdir, paper), rotated)


if __name__ == "__main__":
    #generate()

    # srcdir = 'F:/exam_dataset/waibu/waibu-liangdiangeili'
    srcdir = '../result/lean_biao'

    # outdir = '../result/rotate'
    outdir = '../result/lean_biao_hough'
    test_correct_skew(srcdir,outdir)

    # test_crop_vertical_stipe()