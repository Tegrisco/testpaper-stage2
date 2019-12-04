#!/usr/bin/env python

'''
图像工具
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
import os

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


def preprocess_bw_inv(gray, smooth=True, thresh=200, morph=True, boxSize=(3,3), adaptation=True):

    if smooth:
        gray = cv2.boxFilter(gray, -1, boxSize)

    if adaptation:
        # bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 20)
    else:
        ret, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        # ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if morph:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, element)

    # cv2.imshow('image1', bw2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # bw2[bw2 == 255] = 1
    return bw


def preprocess_bw(gray, smooth=True, thresh=200, morph=True, boxSize=(3,3), adaptation=True):

    if smooth:
        gray = cv2.boxFilter(gray, -1, boxSize)

    if adaptation:
        # bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 20)
    else:
        ret, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        # ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if morph:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, element)

    # cv2.imshow('image1', bw2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # bw2[bw2 == 255] = 1
    return bw


def first_any(arr, axis, target, invalid_val=-1):
    # 第一个任意值的位置
    mask = arr==target
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def vertical_proj(bw):
    """
    weight 是每一列的重量，profile 是每一列第一个非零点到最后一个非零点的距离
    profile -1 表示该列没有白点
    :param bw:
    :return: weight, profile
    """
    # img: black white
    [h,_] = bw.shape

    # 一个像素点权重算1，不算255
    bw = bw//255
    weight = np.sum(bw,axis=0)
    nz1 = first_nonzero(bw,0,-1)
    bw2ud = np.flipud(bw)
    nz2 = h-1-first_nonzero(bw2ud, 0, -1)
    nz2[ nz2 == h ] = -1
    profile = nz2-nz1+1
    profile[ nz1 == -1 ] = -1

    profile = profile.astype(np.float32)
    profile = cv2.GaussianBlur(profile, (1, 3), 0.8)[:,0]
    return weight, profile


def nz_analysis(x):
    """
    分析非零的起始和长度
    x = np.array((0,0,1,1,1,0,0,1,1,0,0))
    返回  (array([2, 7]), array([3, 2]))
    :param x:
    :return:
    """
    length = len(x)
    zpos = np.nonzero(x==0)

    nz_span = np.append(zpos,length)-np.append(-1,zpos)-1
    nz_start = np.append(zpos,length) - nz_span

    return nz_start[nz_span!=0], nz_span[nz_span!=0]


# weight 是每一行的重量，profile 是每一行第一个非零点到最后一个非零点的距离
def horizontal_proj(bw):
    return vertical_proj( np.transpose(bw) )


def strip_white_boder(gray):
    bw = preprocess_bw_inv(gray, smooth=True, thresh=200, morph=True)
    _, profile = vertical_proj(bw)
    profile = profile > 0
    sx = first_nonzero(profile, 0)
    ex = len(profile) - first_nonzero(np.flip(profile, axis=0),0)

    _, profile = horizontal_proj(bw)
    profile = profile > 0
    sy = first_nonzero(profile, 0)
    ey = len(profile) - first_nonzero(np.flip(profile, axis=0), 0)

    return gray[sy:ey,sx:ex]


def strip_white_boder2(gray):
    bw = preprocess_bw_inv(gray, smooth=True, thresh=200, morph=True, boxSize=(10,10))
    _, profile = vertical_proj(bw)
    profile = profile > 0
    sx = first_nonzero(profile, 0)
    ex = len(profile) - first_nonzero(np.flip(profile, axis=0),0)

    _, profile = horizontal_proj(bw)
    profile = profile > 0
    sy = first_nonzero(profile, 0)
    ey = len(profile) - first_nonzero(np.flip(profile, axis=0), 0)

    return [sx, sy, ex, ey]




def align_images(im1, im_ref, savematch=False):
    """
    对齐图像
    :param im1: 被对齐的图像，可为彩色或灰度
    :param im_ref: 参考图像，类型必须和 im1 保持一致
    :param savematch: 是否保存特征点匹配图，只能保存最后一次的匹配
    :return:
    """

    # 需要调整下面参数，保证全部试卷可以，又提高速度
    MAX_FEATURES = 1000
    GOOD_MATCH_PERCENT = 0.5

    # Convert images to grayscale
    if len(im1.shape) == 3:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)
    else:
        im1Gray = im1
        im2Gray = im_ref


    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    if savematch:
        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im_ref, keypoints2, matches, None)
        cv2.imwrite("../result/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # h, _ = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)
    h, _ = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC)

    # Use homography
    height, width = im_ref.shape[:2]
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))
    im1Reg = cv2.warpAffine(im1, h, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE )

    return im1Reg, h


def align_images2x2(im, imReference, savematch=False):

    h, w = im.shape[:2]
    href, wref = imReference.shape[:2]

    margin = 40
    im11 = im[0:h // 2+margin, 0:w // 2+margin]
    imReference11 = imReference[0:href // 2+margin, 0:wref // 2+margin]

    im12 = im[0:h // 2+margin:, w // 2-margin:]
    imReference12 = imReference[0:href // 2+margin, wref // 2-margin:]

    im21 = im[h // 2-margin:, 0:w // 2+margin]
    imReference21 = imReference[href // 2-margin:, 0:wref // 2+margin]

    im22 = im[h // 2-margin:, w // 2-margin:]
    imReference22 = imReference[href // 2-margin:, wref // 2-margin:]

    imReg11, _ = align_images(im11, imReference11, savematch)
    imReg12, _ = align_images(im12, imReference12, savematch)
    imReg21, _ = align_images(im21, imReference21, savematch)
    imReg22, _ = align_images(im22, imReference22, savematch)

    imReg = np.empty_like(imReference)
    imReg[0:href // 2, 0:wref // 2] = imReg11[:-margin,:-margin]
    imReg[0:href // 2, wref // 2:] = imReg12[:-margin,margin:]
    imReg[href // 2:, 0:wref // 2] = imReg21[margin:,:-margin]
    imReg[href // 2:, wref // 2:] = imReg22[margin:,margin:]

    return imReg


def get_skew_from_area(image, sx=0, ex=0, sy=0, ey=0):
    """
    利用包含一个区域的非零像素值的一个可旋转矩形框来判断角度，难点是如何确定这样的区域。
    :param image:
    :param sx:
    :param ex: 为0的话，即为图像宽度
    :param sy:
    :param ey: 为0的话，即为图像高度
    :return:
    """
    h, w = image.shape[:2]

    if ex==0:
        ex = w

    if ey == 0:
        ey = h

    head = image[sy:ey, sx:ex]

    # imshow_(head)

    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    return angle


def rotate_image(image, angle):
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return rotated


def imshow_(img, mode=cv2.WINDOW_NORMAL):
    cv2.namedWindow("temp",mode)
    cv2.imshow("temp",img)
    cv2.waitKey()
    cv2.destroyWindow("temp")


def imwrite_(img, index=0, outdir='./tmp', name='tmp'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    file = os.path.join(outdir, name+str(index)+'.jpg')

    cv2.imwrite(file, img)


def integral_count_nonzero(gray, sdepth=cv2.CV_32S):
    """
    计算非零像素点个数的积分图
    :param gray:
    :param sdepth:
    :return:
    """
    bw=np.zeros_like(gray)
    bw[gray!=0] = 1
    sum = cv2.integral(bw, sdepth)
    return sum


def integral_get_box(sum, box):
    """
    给定一个积分图，计算一个box范围的和
    :param sum:
    :param box:
    :return:
    """
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h

    result = sum[y2, x2] + sum[y1, x1] - sum[y2, x1] - sum[y1, x2]

    return result



if __name__ == "__main__":
    # bw = np.ones((100,80))*255
    # bw[0,:]=0
    # bw = bw.astype(np.uint8)
    # final_boxes = [(0,0,3,3),(10,0,4,3)]
    # sum = integral_count_nonzero(bw)
    # for box in final_boxes:
    #     nzcnt = integral_get_box(sum, box)
    #     print(nzcnt)

    x = np.array((0,0,1,1,1,0,0,1,1,0,0))
    print(nz_analysis(x))
