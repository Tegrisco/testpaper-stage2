#!/usr/bin/python3
# coding = UTF-8

"""
Partitioned script for Python3.

Question collect.

试题搜集生成题库。

"""

import numpy as np
import cv2
import os
import time
import json
import re
import shutil
from rtree import index
from PIL import Image

# xml
from xml.dom.minidom import parse

import sys
sys.path.append('../packages/')

# Image processing tools.
import imutils as imu

# Image rectification
from rectify.hough_angle import rectify

# Get connection components.
from blob_kit.blob_analysis import get_no_intersect_boxes, is_within

# Bai Du character recognition api.
from bd_sdk import bd_rec, bd_access

# SVM classifier.
from svm_model_v0_2.sklearn_svm import classifier

# Detect vertical line.
from svm_model_v0_2.detect_vertical_line import detect_vertical_line3


# Titles' sign. 题号标记
# 0~9 -> 0~9, c1~c9 -> chinese num 1~9, A -> dawn, B -> dot
# c0 -> chinese num 10, E -> else character, add -> additional question
TITLES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 
            'A', 'B', 'c0', 'E', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', 'add']

# SVM characters' type. svm字符类别
NUM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CHI_NUM = [10, 11, 12, 13, 14, 15, 16, 17, 18, 21]

# Characters' string.
NUM_STR = '0123456789'
CHI_NUM_STR = '一二三四五六七八九十'
DOT_STR = '、.'


def mark_box(img, coor, color=(0,0,0), trans=False):
    """Use a rectangular box to show what you need.

    Args:
        img: Image object.
        coor: Coordinates of box.

    Returns:
        An image object.

    """
    font=cv2.FONT_HERSHEY_SIMPLEX
    if trans:
        coor = [length2coor(c) for c in coor]
    for c in coor:
        c = [int(i) for i in c]
        if len(c) == 2:
            cv2.rectangle(img, (c[0],10), (c[1],img.shape[0]-10), color, 2)
        elif len(c) >= 4:
            cv2.rectangle(img, (c[0],c[1]), (c[2],c[3]), color, 2)
            if len(c) == 5:
                cv2.putText(img, TITLES[c[4]], (c[0],c[1]), font, 1.2, (0,0,255), 2)
    return img


def resize(img, w=0, h=0, x=0, y=0):
    """Resize image.Interpolation: cv2.INTER_NEAREST/cv2.INTER_LINEAR
    /INTER_AREA/INTER_CUBIC/INTER_LANCZOS4

    Args:
        img: Source image.
        w: Target image width.
        h: Target image height.
        x: Ratio of x.
        y: Ratio of y.

    Returns:
        An image object.

    """
    if w != 0:
        h = w if h == 0 else h
        i = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
        return cv2.resize(img, (w, h), interpolation=i)
    elif x != 0:
        y = x if y == 0 else y
        i = cv2.INTER_AREA if x < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, (0, 0), fx=x, fy=y, interpolation=i)


def median(data):
    """Get median of an number array.

    Args:
        data: An number array.

    Returns:
        A number.
    """
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half])/2


def take_y(elem):
    return elem[0][0][1]


def take_key(elem):
    return elem[0][1][1]*1000+elem[0][1][0]


def dic2length(dic):
    return (dic["left"], dic["top"], dic["width"], dic["height"])


def dic2coor(dic):
    return (dic["left"], dic["top"], dic["left"]+dic["width"], 
            dic["top"]+dic["height"])


def length2dic(tup):
    return {"left": tup[0], "top": tup[1], "width": tup[2], "height": tup[3]}


def coor2dic(coor):
    return {"left": coor[0], "top": coor[1], "width": coor[2]-coor[0], 
            "height": coor[3]-coor[1]}


def length2coor(tup):
    return (tup[0], tup[1], tup[0]+tup[2], tup[1]+tup[3])


def coor2length(coor):
    return (coor[0], coor[1], coor[2]-coor[0], coor[3]-coor[1])


def num2int(list):
    return [int(e) for e in list]


def get_vertical_white_edge_dis(img):
    """Get vertical white edge distance of character image.

    Args:
        img: Character image.

    Returns:
        An interger distance.
    """
    bw = imu.preprocess_bw_inv(img)
    weight, profile = imu.horizontal_proj(bw)
    horCoorY1 = 0
    if weight[0] == 0:
        i = 0
        while weight[i] == 0 and i < len(weight):
            i += 1
        horCoorY1 = i

    horCoorY2 = len(weight) - 1 
    if weight[-1] == 0:
        j = len(weight) - 1
        while weight[j] == 0 and j > 0:
            j -= 1
        horCoorY2 = j

    return horCoorY1, len(weight)-horCoorY2


def expand_box(box, times=0.8):
    paddingH = heightOfLine*times
    paddingV = heightOfLine*times*0.5
    return (box[0]-paddingH, box[1]-paddingV, box[2]+paddingH+paddingH, box[3]+paddingV+paddingV)


def fuse_box(boxes):
    left = min([b[0] for b in boxes])
    top = min([b[1] for b in boxes])
    right = max([b[2] for b in boxes])
    buttom = max([b[3] for b in boxes])
    box = (left, top, right, buttom)
    return coor2length(box)


def eli_ver(img):
    """Eliminate vertical lines which are longer than 100pixel.

    Args:
        img: Gray image object.

    Returns:
        An gray image.
    """
    binImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 15, -2)
    # cv2.imwrite('binImg.jpg', binImg)
    # exit(0)
    # binImg = img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    dest = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 100))
    dilation = cv2.dilate(dest, element, iterations = 2)

    return cv2.bitwise_or(img, dilation)


def eli_large_cc(coor, img, heightOfLine):
    """Eliminate connection character which is bigger than 
        2 times square of heightOfLine.

    Args:
        imgPath: Inversion image path.
        img: Image object to be eliminated.
        heightOfLine: The median height of lines.
    
    Returns:
        An image.
    """
    for c in coor:
        if c[2]*c[3] > heightOfLine*heightOfLine*2:
            img[c[1]:c[1]+c[3], c[0]:c[0]+c[2]] = 255
        # cv2.rectangle(img, (c[0],c[1]), (c[0]+c[2],c[1]+c[3]), (0,0,255), 2)
    return img


def divide_ver(img):
    """Divide paper by blank detection in vertical direction.

    Args:
        img: Image to be divided.

    Returns:
        A list of horizontal coordinates.
    """
    verCoorX1 = []  
    verCoorX2 = []
    temp = []
    temp1 = []
    temp2 = []
    result = []
    bw = imu.preprocess_bw_inv(img)
    # cv2.imwrite('0001_bw_gaussian.jpg', bw)
    # exit(0)
    #垂直投影
    weight, profile = imu.vertical_proj(bw)
    # print('weight = {}'.format(weight[:100]))
    for i in range(0, len(weight)):
        if weight[i] > 0:
            if i == len(weight)-1 or weight[i+1] == 0:
                verCoorX2.append(i)
            if i == 0 or weight[i-1] == 0:
                verCoorX1.append(i)

    # print(verCoorX1)
    # print(verCoorX2)
    temp = list(zip(verCoorX1, verCoorX2))
    # print(temp)
    
    # for r in temp:
    #     if r[1]-r[0] > 5:
    #         temp1.append(r)
    temp1 = temp

    # first = 1 if temp1[0][0] < 100 else 0
    first = 0
    start = temp1[first][0]
    end = temp1[first][1] 
    for r in range(first, len(temp1)):
        if r == len(temp1)-1:
            end = temp1[r][1]
            temp2.append((start, end))
        elif temp1[r+1][0]-temp1[r][1] < 45:
            continue
        else:
            end = temp1[r][1]
            temp2.append((start, end))
            start = temp1[r+1][0]
    
    for i in temp2:
        if i[1] - i[0] > 800:
            result.append(i)

    if len(result) > 2:
        result = result[0:2]
    # print(result)
    # print(temp)
    return result
    

def divide_hor(img, coor):
    """Divide paper by blank detection in horizontal direction.

    Args:
        img: Image to be divided.

    Returns:
        A list of vertical coordinates.
    """
    h,w = img.shape[0:2]
    result = []
    for c in coor:
        horCoorY1 = []
        horCoorY2 = []
        imgPart = img[0:h, c[0]:c[1]]
        bw = imu.preprocess_bw_inv(imgPart)
        weight, profile = imu.horizontal_proj(bw)
        if weight[0] > 0:
            horCoorY1.append(0)

        for i in range(1, len(weight)-1):
            if weight[i] > 0:
                if weight[i+1] == 0:
                    horCoorY2.append(i)
                if weight[i-1] == 0:
                    horCoorY1.append(i)

        if weight[-1] > 0:
            horCoorY2.append(len(weight)-1)

        # print(verCoorX1)
        # print(verCoorX2)
        line = list(zip(horCoorY1, horCoorY2))
        for l in line:
            if l[1] - l[0] > 10:
                result.append((c[0], l[0], c[1], l[1]))

    return result


def find_char(img, coor):
    """Divide line by blank detection in horizontal direction from left to right.

    Args:
        img: Image to be divided.
        coor: A list of line coordinates.

    Returns:
        A list of first 2 or 3 characters coordinates of each lines.
    """
    result = []
    for i,c in enumerate(coor):
        verCoorX1 = []
        verCoorX2 = []
        line = []
        imgLine = img[c[1]:c[3], c[0]:c[2]]
        bw = imu.preprocess_bw_inv(imgLine, smooth = False)
        # cv2.imwrite('./res/lines/'+str(i)+'.png', bw)
        weight, profile = imu.vertical_proj(bw)
        if weight[0] > 1:
            verCoorX1.append(0)

        for i in range(1, len(weight)-1):
            if weight[i] > 1:
                if weight[i+1] <= 1:
                    verCoorX2.append(i)
                if weight[i-1] <= 1:
                    verCoorX1.append(i)

        if weight[-1] > 1:
            verCoorX2.append(len(weight)-1)

        # print(verCoorX1)
        # print(verCoorX2)
        ver = list(zip(verCoorX1, verCoorX2))
        n = 3
        for v in ver:
            if n == 0:
                break
            elif v[1] - v[0] < 3:
                continue
            else:
                line.append((v[0]+c[0], c[1], v[1]+c[0], c[3]))
                n -= 1
        result.append(line)
    return result


def svm_classify(img, coor):
    """Use svm model to classify character image.

    Args:
        img: Image object with characters.
        coor: Coordinate of characters.

    Returns:
        A list of result.
    """
    result = []
    for n,l in enumerate(coor):
        # labels = []
        tmp = []
        if len(l) < 2:
            continue
        for m,c in enumerate(l):
            if c[2]-c[0] > 100:
                continue

            svmClassifier = classifier()
            # label = svmClassifier.predict(gray=img[c[1]:c[3], c[0]:c[2]], \
            #                                 var_thres=0.035)
            label = svmClassifier.predict(gray=img[c[1]:c[3], c[0]:c[2]])
            if label is not None:
                tmp.append(c + (int(label),))
        # print('tmp: {}'.format(tmp))

        if len(tmp) > 1:
            title = get_title_type(img, tmp)
            result.append((tmp, title))
    # print('result: {}'.format(result))
    return result


def update_result(coor1, coor2):
    """Result integration.

    Args:
        coor1: The result of gray image svm classified result.
        coor2: The result of binary image svm classified result.

    Returns:
        A list of result.
    """
    if len(coor1) == len(coor2):
        coor = zip(coor1, coor2)
        for i, c in enumerate(coor):
            if c[0][1] == 0 and c[1][1] != 0:
                coor1[i] = c[1]
    else:
        coorDic2 = {}
        for c in coor2:
            key = c[0][0][1]
            if key not in coorDic2:
                coorDic2[key] = []
            coorDic2[key].append(c)

        for i, c1 in enumerate(coor1):
            key = c1[0][0][1]
            if key not in coorDic2:
                # print(key)
                continue
            else:
                x = c1[0][0][0]
                c2 = [c for c in coorDic2[key] if x-4 <= c[0][0][0] <= x+4]
                # print(i, c1, c2)
                if len(c2) > 0 and c1[1] == 0 and c2[0][1] != 0:
                    coor1[i] = c2[0]
                    # print('******')
    return coor1


def output_stan(coor, name):
    """Generate standard form result.

    Args:
        coor: SVM classifed result.
        name: Image name.

    Return:
        A dic of standard result.
    """
    stan = {}
    stan["id"] = name[:-4]
    # stan["id"] = os.path.basename(name)
    stan["content"] = []
    for c in coor:
        stan["content"].append({"charsType": [int(x[4]) for x in c[0]], \
                                "location": c[0][0][0:4], "label": c[1]})
    # ([(1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)], 0)
    return stan


def additional_questions(src):
    """Add additional question.

    Args:
        src: A string of BaiDu recognized result.

    Returns:
        A tuple result of additional question or None.
    """
    if "words_result" not in src:
        return None
    # dic = json.loads(src)
    dic = src
    for w in dic["words_result"]:
        # print (w["words"])
        if w["words"].find('思考题') != -1 or w["words"].find('附加题') != -1:
            if w["words"][0] not in CHI_NUM_STR:
                l = w["location"]
                return ((l["left"], l["top"], l["left"]+l["width"], \
                            l["top"]+l["height"], 34))
    return None


def detect_self_assessment(src):
    """Detect the 'Self Accessment' region.

    Args:
        src: The Bai Du recoginized result.

    Returns:
        A coordinate of 'Self Accessment' region or None.
    """
    if "words_result" not in src:
        return None
    # dic = json.loads(src)
    dic = src
    for w in dic["words_result"]:
        # print (w["words"])
        if w["words"].find('自我评估') != -1:
            return w["location"]["top"]
    return None


def get_title_type(img, l):
    """According to first three characters' lable in each line to get 
    the type of title.
    0: No a title.
    1: Xiao Ti.
    2: Da ti.

    Args:
        img: A gray image object.
        l: A list contents first three characters of line.

    Returns:
        An integer of title type.
    """
    title = 0
    if len(l) > 1 and l[0][4] in NUM[1:] and l[1][4] == 20:
        title = 1 if check_position(img, l[0], l[1]) else 0
    elif len(l) > 1 and l[0][4] in CHI_NUM and l[1][4] == 19:
        title = 2 if check_position(img, l[0], l[1]) else 0
    elif len(l) == 3 and l[0][4] in [1,2] and l[1][4] in NUM and l[2][4] == 20:
        title = 1 if check_position(img, l[1], l[2]) else 0
    elif len(l) == 3 and l[0][4] in CHI_NUM and l[2][4] == 19:
        title = 2 if check_position(img, l[1], l[2]) else 0
    return title


def check_position(img, p1, p2):
    """Check two positions of characters.
    
    Args:
        img: A gray image object.
        p1: The position of number character.
        p2: The position of down or dot character.

    Returns:
        A boolean indicates if the positions of characters is reasonable.
    """
    img1 = img[p1[1]:p1[3], p1[0]:p1[2]]
    img2 = img[p2[1]:p2[3], p2[0]:p2[2]]
    im = imu.strip_white_boder(img1)
    t1, b1 = get_vertical_white_edge_dis(img1)
    t2, b2 = get_vertical_white_edge_dis(img2)
    if b2-b1 > im.shape[0]/4 or t2-t1 < im.shape[0]/2 or p2[0]-p1[2] > im.shape[1]:
        return False
    return True


def integrate_lines(coor, lines, size):
    """Integrate lines to region.

    Args:
        coor: The coordinate of titles.
        lines: The coordinate of lines.
        size: Image size.

    Returns:
        The coordinate of region.
    """
    pageWidth = size[1]
    pageHeight = size[0]
    titles = [x for x in coor if x[1] != 0]
    charWidthMax = max([x[0][0][2]-x[0][0][0] for x in titles])
    regionVer = {}

    for x in titles:
        if not regionVer:
            regionVer[x[0][0][:2]] = []
        flag = False
        for r in regionVer:
            if r[0]-charWidthMax*3 < x[0][0][0] < r[0]+charWidthMax*3:
                regionVer[r].append(x)
                flag = True
                break 
        if not flag:
            regionVer[x[0][0][:2]] = [x]
    # print(charWidthMax)
    # print(*regionVer.items(), sep='\n')
    # print(lines)
    # exit(0)
    # partitionLines = []
    keys = list(regionVer.keys())
    keys.sort()
    
    rightMin = min([x[0][0][0] for x in regionVer[keys[1]]]) if len(keys) > 1 else pageWidth
    result = []
    for k,r in regionVer.items():
        ls = [l for l in lines if k[0]-charWidthMax*3 < l[0] < k[0]+charWidthMax*3]
        r.sort(key=take_y)
        rightBound = rightMin - charWidthMax if k == keys[0] else pageWidth - charWidthMax
        if not ls:
            # print(r)
            continue
        startLine = 0
        endLine = 0
        for i,c in enumerate(r):
            temp = []
            if c[1] == 1:
                if i == 0 and c[0][0][1] - ls[marginTop][1] > heightOfLine:
                    # if ls[marginTop][0] != c[0][0][1]:
                    # result.append((ls[marginTop][0], ls[marginTop][1], rightBound, c[0][0][1]-10))
                    result.append((ls[0][0], ls[marginTop][1], rightBound, c[0][0][1]-20))
                startLine = c[0][0][1]
                if i+1 == len(r):
                    # buttom = 
                    endLine = pageHeight - marginButtom if marginButtom != 0 else pageHeight - heightOfLine
                else:
                    endLine = r[i+1][0][0][1] - 20
                # temp = [l for l in ls if startLine <= l[3] <= endLine]
                
            elif c[1] == 2:
                if i == 0 and c[0][0][4] != 10 and c[0][0][1] - ls[marginTop][1] > heightOfLine:
                    # if c[0][0][1]-10-ls[marginTop][1] > heightOfLine:
                    result.append((ls[0][0], ls[marginTop][1], rightBound, c[0][0][1]-20))
                if i+1 == len(r):
                    startLine = c[0][0][1]
                    endLine = pageHeight - marginButtom if marginButtom != 0 else pageHeight - heightOfLine
                elif r[i+1][1] == 2:
                    startLine = c[0][0][1]
                    endLine = r[i+1][0][0][1] - 20
                else:
                    continue
                # temp = [l for l in ls if startLine <= l[3] <= endLine]
            # if temp:
                # result.append((temp[0][0], temp[0][1], temp[0][2], temp[-1][3]))
            result.append((ls[0][0], startLine-10, rightBound, int(endLine)))

    return result


def get_items_v02(region, image, coorCC, name):
    """Gets items from regions.

    Args:
        region: The coordinate of region.
        image: Image data.
        coorCC: The coordinate of connected components.
        name: Image name.

    Returns:
        A dictionary of result.
    """
    result = {}
    result["paper_name"] = name
    result["questions"] = []
    itemDir = os.path.join(desDir, os.path.splitext(name)[0])
    suffix = os.path.splitext(name)[1]
    if os.path.exists(itemDir):
        # os.removedirs(itemDir)
        shutil.rmtree(itemDir)
    os.mkdir(itemDir)


    for i,r in enumerate(region):
        question = {}
        r = (r[0], int(r[1]-heightOfLine*0.2), r[2], r[3])
        if r[2]-r[0] < 15 or r[3]-r[1] < 15:
            continue
        imageRegion = image[r[1]:r[3], r[0]:r[2]]
        coorCCRegion = [coor for coor in coorCC if is_within(coor, coor2length(r))]
        coorCCRegion = [(c[0]-r[0], c[1]-r[1], c[2], c[3]) for c in coorCCRegion]
        # question = get_items_from_region(r, imageRegion, coorCCRegion, itemDir, suffix, idGraph)
        index1, index2 = get_index_from_region(r, imageRegion, coorCCRegion, suffix)
        question["index_out_graph"] = index1
        question["index_in_graph"] = index2
        savePath = os.path.join(itemDir, str(i)+'.png')
        question["question_image"] = savePath
        cv2.imwrite(savePath, imageRegion)
        result["questions"].append(question)
    return result


def get_index_from_region(coorRegion, imageRegion, coorCCRegion, suffix):
    """Gets index from region.

    Args:
        coorRegion: Coordinate of region.
        imageRegion: The image data of region.
        coorCCRegion: The coordinate of connected components in region.
        suffix: Image name's suffix.

    Returns:
        Index out graph and Index in graph.
    """
    bdResult = {}
    index1 = []
    index2 = []
    bdResult = bd_rec(imageRegion, client, imgForm=suffix, api='accurate')
    if "words_result" not in bdResult:
        print(bdResult)
        print("BaiDu recognized error, exit...")
        exit(0)
    if len(bdResult["words_result"]) == 0:
        return index1, index2

    words = []
    for j,w in enumerate(bdResult["words_result"]):
        words.append((j, dic2length(w["location"])))

    graphs = []
    for j,c in enumerate(coorCCRegion):
        if c[2] > heightOfLine*3 or c[3] > heightOfLine*1.5:
        # if c[2] > heightOfLine*0.5 or c[3] > heightOfLine*0.5:
            graphs.append(c)

    for w in words:
        flag = False
        for g in graphs:
            g = expand_box(g)
            if is_within(w[1], g):
                index2.append(bdResult["words_result"][w[0]]["words"])
                flag = True
                break
        if not flag:
            index1.append(bdResult["words_result"][w[0]]["words"])
    return index1, index2



def testpaper(file_path, save_path):
    """Process a paper image.

    Args:
        file_path: File path of image.
        save_path: Destination path.

    Returns:
        Final result.
    """
    name = os.path.basename(file_path)
    image = cv2.imread(file_path)
    if image is None:
        print('ImageError: Could not read the image file, exit...')
        return None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 根据图像长宽判断是否为单栏
    ori_size = image_gray.shape[0:2]
    single_column = True if ori_size[0] > ori_size[1] else False

    # 缩放到固定大小(双栏宽度为4000pixels, 单栏为2000pixels)
    if single_column and ori_size[1] != 2000:
        ratio = 2000/ori_size[1]
    elif not single_column and ori_size[1] != 4000:
        ratio = 4000/ori_size[1]
    else:
        ratio = 0

    if ratio:
        image_gray_scale = resize(image_gray, x=ratio)
    new_size = image_gray_scale.shape[0:2]

    img_bw_inv = imu.preprocess_bw_inv(image_gray_scale)
    img_bw = imu.preprocess_bw(image_gray_scale)
    img_bw2 = imu.preprocess_bw(image_gray_scale, boxSize=(5,5))

    #原始图像去除竖线（长度大于100pixels，可能去除部分题目信息）
    imgEliVer = eli_ver(img_bw2)
    
    #纵向划分试题区域
    coor1 = divide_ver(imgEliVer)
    if not coor1:
        print('coor1: {}'.format(coor1))
        print("Can't divide vertically, skip...")
        return None

    #计算行高（所有行高的中位数）
    coor2 = divide_hor(imgEliVer, coor1)
    if coor2:
        heightOfLines = [c[3]-c[1] for c in coor2]
        heightOfLine = median(heightOfLines)
    else:
        print("Can't divide by line, skip...")
        return None

    #获取原始图像联通区域以及消去较大的联通区域
    imgInv = imu.preprocess_bw_inv(img_bw2.copy())
    cv2.imwrite('imgInv.jpg', imgInv)
    coorCC = get_no_intersect_boxes('imgInv.jpg')
    imgEliCC = eli_large_cc(coorCC, img_bw2.copy(), heightOfLine)
    os.remove('imgInv.jpg')

    #去除较大联通区域的图像上去除竖线
    imgEli = eli_ver(imgEliCC) 

    #行划分
    coor2 = divide_hor(imgEli, coor1)

    #获取每行前三个字符
    coor3 = find_char(img_bw, coor2)

    #SVM分类字符图像
    coor4 = svm_classify(img_bw, coor3)

    #在二值图上重新获取字符以及SVM分类（补全部分漏检）
    imgBw = imu.preprocess_bw(imgEli, boxSize=(4,4), morph=False)
    coor3Bw = find_char(imgBw, coor2)
    coor4Bw = svm_classify(imgBw, coor3Bw)
    coor5 = update_result(coor4, coor4Bw)

    # 百度文字识别API客户端
    client = bd_access()
    try:
        bdResult = bd_rec(imgEliCC, client, imgForm=name[-4:], api='general')
    except Exception as e:
        print(e, 'Baidu recognition error, exit...', sep='\n')
        return None

    # 附加题处理
    addResult = additional_questions(bdResult)
    if addResult:
        coor5.append(([addResult], 2))

    #自我评测处理
    sa = detect_self_assessment(bdResult)
    if sa:
        image_gray_scale[sa:, coor1[1][0]:coor1[1][1]+10] = 255
    coor22 = divide_hor(image_gray_scale, coor1)

    # 生成题目区域
    region = integrate_lines(coor5, coor22, new_size)
    region = [length2coor(list(map(lambda x: int(x / ratio), 
                coor2length(r)))) for r in region]
    
    #背景标记
    region_close = []
    for r in region:
        if r[2] <= r[0] or r[3] <= r[1]: continue
        border = imu.strip_white_boder2(image_gray[r[1]:r[3], r[0]:r[2]])
        r = num2int([border[0]+r[0], border[1]+r[1], border[2]+r[0], border[3]+r[1]])
        region_close.append(r)
        background = np.zeros((r[3]-r[1], r[2]-r[0], 3), np.uint8)
        background[:] = (200,170,150)
        image[r[1]:r[3], r[0]:r[2]] = cv2.addWeighted(image[r[1]:r[3], 
                                        r[0]:r[2]], 0.7, background, 0.3, 0)

    # image_process = np.concatenate([image_gray_scale, img_bw, 
    #                 imgEliVer, imgEliCC, imgEli], axis=0)
    # cv2.imwrite("image_process.jpg", image_process)
    dic_result = {"positions": region_close}
    print(f'region: {region_close}')
    print ('output to directory {}'.format(save_path))
    with open(save_path+'/'+name.replace('.jpg', '.json'), 'w') as f:
        json.dump(dic_result, f)
    cv2.imwrite(os.path.join(save_path, name), image)
    return region

    #去除装订线及中间分割线（不会去除题目信息）
    imgData1 = detect_vertical_line3(imgData.copy())

    #导出最后结果
    resultFinal = get_items_v02(region, imgData1.copy(), coorCC, name)
    if desPath:
        with open(desPath[:-4]+'.json', 'w') as f:
            json.dump(resultFinal, f)

    return resultFinal
    


# Initialisation starts --------------------------------------------

keyWordsConfig = '../configs/key_words.txt'
heightOfLine = 0
marginButtom = 0
marginTop = 0
client = ''
keyWords = ''
xmlPath = ''
now = time.strftime('%m-%d-%H-%M-%S', time.localtime())

def init():
    """Initialization function"""
    global keyWords, marginTop, marginButtom, client

    if not os.path.isdir(srcDir):
        print('Direction error! exit...')
        exit(0)

    with open(keyWordsConfig,'r', encoding='UTF-8') as f:
        for i in f.readlines():
            keyWords = keyWords + str(i.strip())
            keyWords = keyWords + "|"
    keyWords = keyWords[:-1]

    if xmlPath:
        DOMTree=parse(xmlPath)
        paperlist=DOMTree.documentElement
        papers=paperlist.getElementsByTagName('paper')
        for p in papers:
            if p.getAttribute('category') == os.path.basename(srcDir):
                marginButtom = int(p.getElementsByTagName('marginButtom')[0].childNodes[0].data)
                marginTop = int(p.getElementsByTagName('marginTop')[0].childNodes[0].data)
                break

    client = bd_access()


# Initialisation ends ---------------------------------------------



# Main starts --------------------------------------------
if __name__ == '__main__':
    print('Extract questions on papers...')
    print('Usage example: python3 ti_div_v06.py XXX.jpg [target_directory]')

    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
        print(f'Image File {file_path} to be processed...')
        if len(sys.argv) >= 3:
            save_path = sys.argv[2]
        else:
            save_path = os.path.split(file_path)[0]
    else:
        print('no valid input argument, exit...')
        sys.exit()

    start = time.process_time()
    testpaper(file_path, save_path)
    end = time.process_time()
    print('Costs time: %ss'%(end-start))
# Main ends --------------------------------------------




