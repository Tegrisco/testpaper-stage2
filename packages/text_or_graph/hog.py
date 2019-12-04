#!/usr/bin/env python

import cv2
import numpy as np

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(20, 20),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img
## [deskew]

## [hog]
def hog(img, bin_n=16):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
## [hog]

def hog2x2(img, nbin=16):
    """

    :param img: 尺寸需为偶数
    :param nbin:
    :return:
    """
    h, w = img.shape
    hh = h//2
    hw = w//2
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(nbin*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:hh,:hw], bins[hh:,:hw], bins[:hh,hw:], bins[hh:,hw:]
    mag_cells = mag[:hh,:hw], mag[hh:,:hw], mag[:hh,hw:], mag[hh:,hw:]
    hists = [np.bincount(b.ravel(), m.ravel(), nbin) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 4*nbin bit vector
    return hist


def hog1x1(img, nbin=16):
    """

    :param img: 尺寸需为偶数
    :param nbin:
    :return:
    """
    h, w = img.shape
    hh = h//2
    hw = w//2
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(nbin*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    hist = np.bincount(bins.ravel(), mag.ravel(), nbin)
    return hist


def hog_2scale(img, nbin=9, h=32, w=32):

    img = cv2.resize(img, (h, w))
    hh = h // 2
    hw = w // 2
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(nbin * ang / (2 * np.pi))  # quantizing binvalues in (0...16)

    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(nbin * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    hist1 = np.bincount(bins.ravel(), mag.ravel(), nbin)

    bin_cells = bins[:hh, :hw], bins[hh:, :hw], bins[:hh, hw:], bins[hh:, hw:]
    mag_cells = mag[:hh, :hw], mag[hh:, :hw], mag[:hh, hw:], mag[hh:, hw:]
    hists = [np.bincount(b.ravel(), m.ravel(), nbin) for b, m in zip(bin_cells, mag_cells)]
    hist2 = np.hstack(hists)  # hist is a 4*nbin bit vector

    return np.hstack((hist1, hist2))





def hog_svm():
    bin_n = 16  # Number of bins
    img = cv2.imread('./tmp/digits.png', 0)
    if img is None:
        raise Exception("we need the digits.png image from samples/data here !")

    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

    # First half is trainData, remaining is testData
    train_cells = [i[:50] for i in cells]
    test_cells = [i[50:] for i in cells]

    ######     Now training      ########################

    deskewed = [list(map(deskew, row)) for row in train_cells]
    hogdata = [list(map(hog_2scale, row)) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1, 45)
    responses = np.repeat(np.arange(10), 250)[:, np.newaxis]

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('./tmp/svm_data.dat')

    ######     Now testing      ########################

    deskewed = [list(map(deskew, row)) for row in test_cells]
    hogdata = [list(map(hog_2scale, row)) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, 45)
    result = svm.predict(testData)[1]

    #######   Check Accuracy   ########################
    mask = result == responses
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)


if __name__ == "__main__":
    # hog_svm()
    import os
    from src.base import data_dir

    srcdir = os.path.join(data_dir, 'waibu/jiangsumijuan')


    # papers = os.listdir(srcdir)
    papers = ['120190703153513729.jpg']

    for paper in papers:
        if paper.endswith('.jpg'):
            imgfile = os.path.join(srcdir, paper)

            print(imgfile)

            img = cv2.imread(imgfile)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            x = 100
            y = 200
            w = 40
            h = 30
            box = (x, y, w, h)

            hog_feature = hog_2scale(img[y:y+h,x:x+w])
            print(hog_feature)
