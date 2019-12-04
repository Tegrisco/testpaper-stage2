from skimage.feature import corner_harris, corner_peaks,greycomatrix
import cv2
import numpy as np
import math

import sys
sys.path.append('../packages/')
from text_or_graph.hog import hog_2scale
from blob_kit.blob_analysis import preprocess_bw_inv, get_no_intersect_boxes

def integral_get_box(sum, box):

    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    result = sum[y2, x2] + sum[y1, x1] - sum[y2, x1] - sum[y1, x2]

    return result


def build_filters():
    filters = []
    ksize = 13  # gabor尺度
    lamda = np.pi / 2.0  # 波长
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor方向，0°，22.5,45°，67.5,90°，112.5,135°，共8个
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        # getGaborKernal(Ksize(卷积核大小），sigma（高斯方差），theta（角度），lamda（波长），gamma（纵横比），psi（相位差），ktype（mat数据类型））
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    # dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    dots = np.where(x.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rlet(x):
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    # dots = np.where(x.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def process_img(gray):
    bw2 = preprocess_bw_inv(gray)
    # bw2:二值化图片
    bw2[bw2 != 0] = 1
    sum = cv2.integral(bw2, sdepth=cv2.CV_64F)
    #sum:积分图
    coords = corner_peaks(corner_harris(bw2), min_distance=1)
    #coords：角点坐标
    filters = build_filters() #filters:滤波器
    # res=process_threaded(img, filters, threadn = 8)
    res = []  # 滤波结果
    for kern in filters:
        fimg = cv2.filter2D(gray, cv2.CV_8UC1, kern)
        res.append(fimg)
    sum_oldlist = []
    sum_newlist = []
    for temp in range(len(res)):
        fimg_gray = res[temp].astype(np.float32)
        fimg_square = np.multiply(np.array(fimg_gray), np.array(fimg_gray))
        sum_old = cv2.integral(res[temp], sdepth=cv2.CV_64F)
        # 滤波图片积分图
        sum_new = cv2.integral(fimg_square, sdepth=cv2.CV_64F)
        # 滤波图片平方的积分图
        sum_oldlist.append(sum_old)   # 8张滤波结果的积分图列表
        sum_newlist.append(sum_new)   #8张滤波结果平方后的积分图列表
    return sum, sum_oldlist, sum_newlist, bw2, coords


def get_ccfeature(img,gray,boxes):
    sum, sum_oldlist, sum_newlist, bw2, coords = process_img(gray)
    feature_boxes = []
    data = np.empty(shape=[1, 94])
    for box in boxes:
        x, y, w, h = box
        # x, y, w, h= box
        # print(x, y, w, h, label)
        # cv2.rectangle(bw2, (x, y), (x + w, y + h), (0, 0, 255), 2);imu.imshow_(gray)

        # height,width,area,and aspect ratio of the bonding box of CC
        s = w * h
        aspect_ratio = h / w

        # pixel number,duty factor of cc
        pixel_number = integral_get_box(sum, box)
        duty_factor = pixel_number / s

        # Harris corner point number of CC's image
        box_coord = []
        for coord in coords:
            cy = coord[0]
            cx = coord[1]
            if cx >= x and cx <= x + w and cy >= y and cy <= y + h:
                # cv2.circle(bw2, (cx, cy), 5, (0, 255, 255), 1)
                box_coord.append(coord)
        corners_number = len(box_coord)

        # gabor feature:8 direction ，each direction's mean and standard deviation of filtered gray level
        gabor_mean = []
        sd_box = []
        for i in range(8):
            result = integral_get_box(sum_oldlist[i], box)
            mean = result / s
            # 每个box像素和的均值
            result_square = integral_get_box(sum_newlist[i], box)
            # 滤波图片平方的box积分和
            standard_deviation = math.sqrt((result_square - (result ** 2) / s) / s)

            # mean of filtered gray level,8个
            # standard deviation of filtered gray level，8个
            sd_box.append(standard_deviation)
            gabor_mean.append(mean)
        run_length_box = rle_encoding(bw2[y:y + h, x:x + w])
        # mean_run_length of cc
        mean_run_length = pixel_number / (len(run_length_box) / 2)
        run_length_box_new = run_length_box[1::2]
        run_length_box_n = np.array(run_length_box_new)
        # variance of run length
        variance = run_length_box_n.var()
        run_lengtht_box = rlet(bw2[y:y + h, x:x + w])
        run_length_box_new = run_length_box[1::2]
        run_lengtht_box_new = run_lengtht_box[1::2]
        # 矩阵的run_length，以及horizontal and vertical run length histogram
        max = 0
        for run_length in run_length_box_new:
            if run_length > max:
                max = run_length
        bin1 = 0
        bin2 = 0
        bin3 = 0
        bin4 = 0
        bin5 = 0
        bin6 = 0
        for run_length in run_length_box_new:
            if 0 < run_length <= max / 6:
                bin1 += 1
            if max / 6 < run_length <= max / 3:
                bin2 += 1
            if max / 3 < run_length <= max / 2:
                bin3 += 1
            if max / 2 < run_length <= max * 2 / 3:
                bin4 += 1
            if max * 2 / 3 < run_length <= max * 5 / 6:
                bin5 += 1
            if max * 5 / 6 < run_length <= max:
                bin6 += 1

        # 转置矩阵的run_length，以及horizontal and vertical run length histogram
        maxt = 0
        for run_lengtht in run_lengtht_box_new:
            if run_lengtht > maxt:
                maxt = run_lengtht
        bint1 = 0
        bint2 = 0
        bint3 = 0
        bint4 = 0
        bint5 = 0
        bint6 = 0
        for run_lengtht in run_lengtht_box_new:
            if 0 < run_lengtht <= maxt / 6:
                bint1 += 1
            if maxt / 6 < run_lengtht <= maxt / 3:
                bint2 += 1
            if maxt / 3 < run_lengtht <= maxt / 2:
                bint3 += 1
            if maxt / 2 < run_lengtht <= maxt * 2 / 3:
                bint4 += 1
            if maxt * 2 / 3 < run_lengtht <= maxt * 5 / 6:
                bint5 += 1
            if maxt * 5 / 6 < run_lengtht <= maxt:
                bint6 += 1

        glcm = greycomatrix(bw2[y:y + h, x:x + w], distances=[1, 2, 3, 4, 5, 6], angles=[0, np.pi / 2], levels=2,
                            symmetric=False,
                            normed=False)
        # glcm:4D灰度共生矩阵，[i,j,distance,angles]
        xd1 = glcm[1, 1, 0, 0]
        xd2 = glcm[1, 1, 1, 0]
        xd3 = glcm[1, 1, 2, 0]
        xd4 = glcm[1, 1, 3, 0]
        xd5 = glcm[1, 1, 4, 0]
        xd6 = glcm[1, 1, 5, 0]

        yd1 = glcm[1, 1, 0, 1]
        yd2 = glcm[1, 1, 1, 1]
        yd3 = glcm[1, 1, 2, 1]
        yd4 = glcm[1, 1, 3, 1]
        yd5 = glcm[1, 1, 4, 1]
        yd6 = glcm[1, 1, 5, 1]

        hog_arrray = hog_2scale(img[y:y + h, x:x + w], nbin=9, h=32, w=32)
        hoglist = hog_arrray.tolist()
        featurelist = process_featurelist(h, w, s, aspect_ratio, pixel_number, duty_factor, corners_number,
                                          mean_run_length, variance, bin1, bin2, bin3, bin4, bin5, bin6, bint1, bint2,
                                          bint3, bint4, bint5, bint6, xd1, xd2, xd3, xd4, xd5, xd6, yd1, yd2, yd3, yd4,
                                          yd5, yd6)
        featurelist.extend(gabor_mean)
        featurelist.extend(sd_box)
        featurelist.extend(hoglist)
        data = np.append(data, np.array([featurelist]), axis=0)
        feature_boxes.append(featurelist)
    data = np.delete(data, 0, axis=0)
    print(data.shape)
    return data


def process_featurelist(h, w, s, aspect_ratio, pixel_number, duty_factor, corners_number, mean_run_length, variance, bin1, bin2, bin3, bin4, bin5, bin6, bint1, bint2, bint3, bint4, bint5, bint6, xd1, xd2, xd3, xd4, xd5, xd6, yd1, yd2, yd3, yd4, yd5, yd6):
    featurelist=[]
    featurelist.append(h)
    featurelist.append(w)
    featurelist.append(s)
    featurelist.append(aspect_ratio)
    featurelist.append(pixel_number)
    featurelist.append(duty_factor)
    featurelist.append(corners_number)
    featurelist.append(mean_run_length)
    featurelist.append(variance)
    featurelist.append(bin1)
    featurelist.append(bin2)
    featurelist.append(bin3)
    featurelist.append(bin4)
    featurelist.append(bin5)
    featurelist.append(bin6)
    featurelist.append(bint1)
    featurelist.append(bint2)
    featurelist.append(bint3)
    featurelist.append(bint4)
    featurelist.append(bint5)
    featurelist.append(bint6)
    featurelist.append(xd1)
    featurelist.append(xd2)
    featurelist.append(xd3)
    featurelist.append(xd4)
    featurelist.append(xd5)
    featurelist.append(xd6)
    featurelist.append(yd1)
    featurelist.append(yd2)
    featurelist.append(yd3)
    featurelist.append(yd4)
    featurelist.append(yd5)
    featurelist.append(yd6)
    return featurelist

if __name__ == '__main__':
    src_imgpath = r'../img/2.jpg'
    img = cv2.imread(src_imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw2 = preprocess_bw_inv(gray)
    cv2.imwrite('../result/xxx.jpg', bw2)
    black_imgpath = '../result/xxx.jpg'
    boxes=get_no_intersect_boxes(black_imgpath)
    # boxesFile=open(r'..\EasyLabel\EasyLabel\output\120190703153513729.txt', 'r')
    # # boxes=eval(open(r'..\EasyLabel\EasyLabel\output\120190703153513729.txt','r'))
    # # # box=line.readline()
    # # # line2=line.readline()
    # # # print(box)
    # # # print(line2)
    # boxes = []
    # # print()
    # for box in boxesFile.readlines():
    #     boxes.append(eval(box))
    # #
    # # for box in boxes:
    # #     print(box[0])
    data = get_ccfeature(img, gray, boxes)