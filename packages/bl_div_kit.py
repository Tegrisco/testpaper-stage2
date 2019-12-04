#!/usr/bin/python3
# coding = UTF-8

'''
Simple partitioned script for Python3.

'''

import numpy as np
import cv2
import os
from PIL import Image

import sys
sys.path.append('../packages/')
import imutils as imu


def draw_line(img, coor, color = (0, 255, 0)):
    for c in coor:
        cv2.line(img, (c,0), (c,img.shape[0]), color, 1)
    return img


def mark_text(img, coor, color=(0,0,0)):
    font=cv2.FONT_HERSHEY_SIMPLEX
    for c in coor:
        if len(c) == 5:
            cv2.putText(img, TITLES[c[4]], (c[0],c[1]), font, 1.2, (0,0,255), 2)
    return img


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


def resize(img, w=0, h=0, x=0, y=0, i=cv2.INTER_NEAREST):
    """Resize image.

    Args:
        img: Source image.
        w: Target image width.
        h: Target image height.
        x: Ratio of x.
        y: Ratio of y.
        i: Interpolation, cv2.INTER_NEAREST/cv2.INTER_LINEAR/INTER_AREA/
                        INTER_CUBIC/INTER_LANCZOS4

    Returns:
        An image object.

    """
    if w != 0:
        return cv2.resize(img, (w, h), interpolation=i)
    elif x != 0:
        return cv2.resize(img, (0, 0), fx=x, fy=y, interpolation=i)


def make_border(img, t=0, b=0, l=0, r=0, borderType=cv2.BORDER_CONSTANT, color=(255,255,255)):
    """Make image border.

    Args:
        img: Source image.
        t: Top border.
        b: Bottom border.
        l: Left border.
        r: Right border.
        borderType: cv2.BORDER_CONSTANT/cv2.BORDER_DEFAULT
        color: Border color.

    Returns:
        An image object.

    """
    return cv2.copyMakeBorder(img, t, b, l, r, borderType, value=color)



def adjust_img(img):
    """Adjust image to 28*28pixel.

    Args:
        img: Source image.

    Returns:
        An image object.

    """
    width = img.shape[1]
    height = img.shape[0]
    ratio = 28/width if width > height else 28/height
    try:
        img = resize(img, x=ratio, y=ratio)
    except cv2.error as e:
        # print('There is an error occurred when scaling this image!')
        # print('width= {}, height= {}, ratio= {}'.format(width, height, ratio))
        # cv2.imshow('char', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('Force resize to 28*28pixel...')
        img = resize(img, w=28, h=28)
    w = img.shape[1]
    h = img.shape[0]
    dif = int(abs(w-h)/2)
    wb = dif if w != 28 else 0
    hb = dif if h != 28 else 0
    img = make_border(img, t=hb, b=hb, l=wb, r=wb)    
    if img.shape[0] != 28:
        img = make_border(img, b=1)
    elif img.shape[1] != 28:
        img = make_border(img, r=1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img



