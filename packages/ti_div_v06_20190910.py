#!/usr/bin/python3
# coding = UTF-8

'''
Partitioned script for Python3.

 Question collect.

试题搜集生成题库。

'''

import numpy as np
import cv2
import os
import time
import json
import pickle
import re
import shutil
from rtree import index
from PIL import Image
import matplotlib.pyplot as plt

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

# Text and graph classifier.
from text_or_graph.text_or_graph import tell_me_text_or_graph


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


class Paper():
    """docstring for Paper"""
    name = ''
    image = []
    coor = []
    datis = []
    xiaotis = []
    valid = True

    def __init__(self, name, image, coor):
        self.name = name
        self.image = image
        self.coor = coor
        self.datis = []
        self.xiaotis = []
        self.valid = True

    def take_y(self, elem):
        if elem[0] > self.image.shape[1]/3:
            return elem[1] + self.image.shape[0]
        else:
            return elem[1]


    def generate(self):
        self.coor.sort(key=self.take_y)
        # print(self.coor)
        dati = 0
        keysOfDati = []
        for c in self.coor:
            if c[4] in CHI_NUM:
                dati = c[4]
                keysOfDati.append(dati)
                self.datis.append({"title": TITLES[dati], "location": c[0:4],"content": []})
            else:
                xiaoti = c[4] - 13 if c[4] >= 23 else c[4]
                self.xiaotis.append({"title": xiaoti, "location": c[0:4], "belong": TITLES[dati]})
                if len(self.datis) == 0:
                    self.datis.append({"title": TITLES[0], "location": None,"content": []})
                self.datis[-1]["content"].append(xiaoti)

        if not is_ordered(keysOfDati):
            self.valid = False
        else:
            for d in self.datis:
                if not is_ordered(d["content"]) or (d["title"] != '0' and \
                    len(d["content"]) > 0 and 1 not in d["content"]):
                    self.valid = False
                    break


    def structure(self):
        print('Paper structure:')
        for d in self.datis:
            print('+--{} {}'.format(d["title"], d["location"]))
            xiaotisTmp = [x for x in self.xiaotis if x["belong"]==d["title"]]
            for x in xiaotisTmp:
                print('|    +--{} {}'.format(x["title"], x["location"]))
        print('valid: {}'.format(self.valid))


def is_ordered(array):
    """Determine whether an array is ordered。

    Args:
        array: An integer array.

    Returns:
        A boolean.
    """
    if 0 in array:
        del array[0]
    for i,a in enumerate(array[:-1]):
        if array[i+1] != a+1:
            return False
    return True


def is_special(title):
    special = False            
    match = re.search(keyWords, title)
    special = True if match else False
    return special


def mark_box(img, coor, color=(0,0,0)):
    """Use a rectangular box to show what you need.

    Args:
        img: Image object.
        coor: Coordinates of box.

    Returns:
        An image object.

    """
    font=cv2.FONT_HERSHEY_SIMPLEX
    for c in coor:
        c = [int(i) for i in c]
        if len(c) == 2:
            cv2.rectangle(img, (c[0],10), (c[1],img.shape[0]-10), color, 2)
        elif len(c) >= 4:
            cv2.rectangle(img, (c[0],c[1]), (c[2],c[3]), color, 2)
            if len(c) == 5:
                cv2.putText(img, TITLES[c[4]], (c[0],c[1]), font, 1.2, (0,0,255), 2)
    return img


def show_image(img):
    image  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.show()


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


def get_dilation(gray):
    bw = imu.preprocess_bw_inv(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (49, 13))
    bw2 = cv2.dilate(bw, kernel, iterations=1)
    return bw2


def get_image_box(gray):
    bw = imu.preprocess_bw_inv(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (49, 13))
    bw2 = cv2.dilate(bw, kernel, iterations=3)
    image, contours, hierarchy = cv2.findContours(bw2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # cv2.imwrite('temp.png', bw2)
    cv2.namedWindow("bw2", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("bw2", bw2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        boxList = box.tolist()
        origin = min(boxList)
        # print(box)

        if(height > 150):
            boxes.append((origin[0], origin[1], origin[0]+width, origin[1]+height))
    # exit(0)
    return boxes


def eli_ver(img):
    """Eliminate vertical lines which are longer than 100pixel.

    Args:
        img: Gray image object.

    Returns:
        An gray image.
    """
    binImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 15, -2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    dest = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 100))
    dilation = cv2.dilate(dest, element, iterations = 2)

    return cv2.bitwise_or(img, dilation)


def absolute_coor(dicResult):
    for q in dicResult["questions"]:
        r = q["location"]
        for i in q["content"]:
            i["location"]["left"] += r["left"]
            i["location"]["top"] += r["top"]
    return dicResult


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


def eli_large_cc(coor, img, heightOfLine):
    """Eliminate connection character which is bigger than 
        1.5 times square of heightOfLine.

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


# def eli_title_num(coor, image):


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
    
    for r in temp:
        if r[1]-r[0] > 5:
            temp1.append(r)

    first = 1 if temp1[0][0] < 100 else 0
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
    # with open(stanPath[:-4]+'.json', 'w') as f:
    #     json.dump(stan, f)

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


def verification(result, stanPath):
    """Verifies the result with standard result.

    Args:
        result: The standard form of reault.
        stanPath: The standard result json file path.

    Returns:
        An integer of mistaken number.
    """
    stanFile = open(stanPath)
    stan = json.load(stanFile)
    mistakes = 0
    if result["id"] != stan["id"]:
        print(result["id"], stan["id"])
        print("Paper not match!")
        return None
    # titles1 = [t in result["content"] if t["label"] != 0]
    if len(result["content"]) == len(stan["content"]):
        for r,s in zip(result["content"], stan["content"]):
            if r["label"] != s["label"]:
                print(r, s)
                mistakes += 1
    
    else:
        #print("Number of lines not match!")
        #print(len(result["content"]), len(stan["content"]))
        #print(*result["content"], sep='\n')
        stanDic = {}
        for c in stan["content"]:
            key = c["location"][1]
            if key not in stanDic:
                stanDic[key] = []
            stanDic[key].append(c)

        for i, c1 in enumerate(result["content"]):
            key = c1["location"][1]
            if key not in stanDic:
                # print(key)
                continue
            else:
                x = c1["location"][0]
                c2 = [c for c in stanDic[key] if x-4 <= c["location"][0] <= x+4]
                # print(i, c1, c2)
                if len(c2) > 0 and  c1["label"] != c2[0]["label"]:
                    print(c1, c2[0])
                    mistakes += 1
    return mistakes


def integrate_lines(coor, lines, pageWidth, pageHeight):
    """
    """
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

    # partitionLines = []
    keys = list(regionVer.keys())
    keys.sort()
    key1 = keys[0]
    key2 = keys[1]
    rightMin = min([x[0][0][0] for x in regionVer[key2]])
    result = []
    for k,r in regionVer.items():
        ls = [l for l in lines if k[0]-charWidthMax*3 < l[0] < k[0]+charWidthMax*3]
        r.sort(key=take_y)
        rightBound = rightMin - charWidthMax if k == key1 else pageWidth - charWidthMax
        if not ls:
            print(r)
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


def get_items(region, image, bdResult, CCAndType, name):
    if "words_result" not in bdResult:
        return None
    result = {}
    result["paper_name"] = name
    result["questions"] = []
    itemDir = os.path.join(desDir, os.path.splitext(name)[0])
    suffix = os.path.splitext(name)[1]
    if os.path.exists(itemDir):
        # os.removedirs(itemDir)
        shutil.rmtree(itemDir)
    os.mkdir(itemDir)

    imageBoxes = [ct[0] for ct in CCAndType if ct[1]==1]
    # for i,r in enumerate(region):
    #     gray = image[r[1]:r[3], r[0]:r[2]]
    #     boxes = get_dilation(gray)
    #     imageBoxes += [(box[0]+r[0], box[1]+r[1], box[2]+r[0], box[3]+r[1]) for box in boxes]


    idx = index.Index()
    lengthText = len(bdResult["words_result"])
    lengthImage = len(imageBoxes)

    for i,w in enumerate(bdResult["words_result"]):
        l = w["location"]
        box = (l["left"], l["top"], l["left"]+l["width"], l["top"]+l["height"])
        if len(w["words"]) < 4:
            continue
        idx.insert(i, box, obj=0)

    for i,b in enumerate(imageBoxes):
        box = (b[0], b[1], b[0]+b[2], b[1]+b[3])
        idx.insert(i+lengthText, box, obj=1)

    for r in region:
        xiaoti = {}
        hits = list(idx.intersection(r, objects=True))
        items = [item for item in hits if r[1] < item.bbox[1]+(item.bbox[3]-item.bbox[1])/2 < r[3]]
        idOfTexts = [item.id for item in items if item.object==0]
        idOfImages = [item.id for item in items if item.object==1]
        xiaoti["title"] = bdResult["words_result"][idOfTexts[0]]["words"] if len(idOfTexts)!=0 else ''
        xiaoti["location"] = {"left": r[0], "top": r[1], "width": r[2]-r[0], "height": r[3]-r[1]}
        xiaoti["content"] = []
        for j,it in enumerate(items):
            item = {}
            c = [int(b) for b in it.bbox]
            item["id"] = j
            item["location"] = {"left": c[0], "top": c[1], "width": c[2]-c[0], "height": c[3]-c[1]}
            if it.object == 0:
                item["type"] = "text"
                words = bdResult["words_result"][it.id]["words"]
                item["value"] = words
            elif it.object == 1:
                item["type"] = "graph"
                itemPath = os.path.join(itemDir, str(it.id)+'.png')
                cv2.imwrite(itemPath, image[c[1]:c[3], c[0]:c[2]])
                item["value"] = itemPath
            xiaoti["content"].append(item)
        result["questions"].append(xiaoti)
        
    return result


def get_items_v02(region, image, coorCC, name):
    """
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

    # idxCC = index.Index()
    # for j,c in enumerate(coorCC):
    #     idxCC.insert(j, length2coor(c))

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


def get_items_from_region(coorRegion, imageRegion, coorCCRegion, itemDir, suffix, idGraph):
    """

    Args:
        coorRegion: The coordinate of region.
        imageRegion: The image of region.
        coorCCRegion: The coordinate of CC in region.
        suffix: Image form suffix.
        idGraph: The id of graph in paper.

    Returns:
        A dic of question result.

    """
    xiaoti = {}
    bdResult = {}
    bdResult = bd_rec(imageRegion, client, imgForm=suffix, api='accurate')
    title = ''
    if "words_result" in bdResult and len(bdResult["words_result"]) > 0:    #结果返回错误处理
        title = bdResult["words_result"][0]["words"]
    # i = 0
    # if "words_result" in bdResult:
    #     while i < len(bdResult["words_result"]) and \
    #         not title.endswith(('。', '？', ')')):
    #         title += bdResult["words_result"][i]["words"]
    #         i += 1

    xiaoti["title"] = title
    xiaoti["location"] = coor2dic(coorRegion)
    xiaoti["content"] = []
        
    if is_special(title):
        locationOfTitle = bdResult["words_result"][0]["location"]
        boxOfGraph = (10, dic2coor(locationOfTitle)[3]+10, coorRegion[2]-coorRegion[0]-10, 
                        coorRegion[3]-coorRegion[1]-10)
        itemText = {}
        itemText["id"] = 0
        itemText["location"] = locationOfTitle
        itemText["type"] = "text"
        itemText["value"] = title
        xiaoti["content"].append(itemText)

        itemGraph = {}
        itemPath = os.path.join(itemDir, str(idGraph)+'.png')
        cv2.imwrite(itemPath, imageRegion[boxOfGraph[1]:boxOfGraph[3], boxOfGraph[0]:boxOfGraph[2]])
        itemGraph["id"] = 1
        itemGraph["location"] = coor2dic(boxOfGraph)
        itemGraph["type"] = "graph"
        itemGraph["value"] = itemPath
        xiaoti["content"].append(itemGraph)
        return xiaoti

    wordBoxes = []
    if "words_result" in bdResult:
        for j,w in enumerate(bdResult["words_result"]):
            wordBoxes.append((j, dic2length(w["location"])))

    graphBoxes = []

    for j,b in wordBoxes:
        # print(j,b)
        b = expand_box(b)
        coorCCRegion = [c for c in coorCCRegion if not is_within(c, b)]
    
    n = 0
    for j,c in enumerate(coorCCRegion):
        # if c[2] > heightOfLine*3 or c[3] > heightOfLine*1.5:
        if c[2] > heightOfLine*0.5 or c[3] > heightOfLine*0.5:
            graphBoxes.append((n,c))
            n += 1
            c = expand_box(c)
            wordBoxes = [it for it in wordBoxes if not is_within(it[1], c)]

    mixBoxes = []
    for j,w in enumerate(wordBoxes.copy()):
        if w[1][2] > len(bdResult["words_result"][w[0]]["words"])*heightOfLine*1.5:
            del(wordBoxes[j-len(mixBoxes)])
            mixBoxes.append((n, w[1]))
            n += 1

    # Fuses image
    graphBoxes += mixBoxes
    idxWords = index.Index()
    for j,b in enumerate(wordBoxes):
        if b[1][2] < heightOfLine*5:
            idxWords.insert(b[0], length2coor(b[1]))
    wordBoxes = [b for b in wordBoxes if b[1][2] >= heightOfLine*5]

    count = 0
    idDeleted = []
    for j,b in graphBoxes.copy():
        bEx = expand_box(b)
        items = list(idxWords.intersection(length2coor(bEx), objects=True))
        if len(items) > 0:
            boxes = [num2int(i.bbox) for i in items] + [length2coor(b)]
            box = fuse_box(boxes)
            for i,it in enumerate(items):
                # del(wordBoxes[it.id-i-len(idDeleted)])
                idxWords.delete(it.id, it.bbox)
                idDeleted.append(it.id)
            del(graphBoxes[j-count])
            count += 1
            graphBoxes.append((n, box))
            n += 1
    if len(idxWords.leaves()[0][1]) > 0:
        wordItems = list(idxWords.intersection(idxWords.get_bounds(), objects=True))
        wordBoxes += [(i.id, coor2length(num2int(i.bbox))) for i in wordItems]

    idxGraphs = index.Index()
    for j,b in enumerate(graphBoxes):
        idxGraphs.insert(j, length2coor(b[1]))

    count = 0
    for j,b in enumerate(graphBoxes.copy()):
        bEx = expand_box(b[1])
        items = list(idxGraphs.intersection(length2coor(bEx), objects=True))
        if len(items) > 0:
            boxes = [num2int(i.bbox) for i in items] + [length2coor(b[1])]
            box = fuse_box(boxes)
            for i,it in enumerate(items):
                # if it.id-i not in range(0, len(graphBoxes)):
                #     print(it.id-i)
                #     print(graphBoxes)
                #     continue
                idxGraphs.delete(it.id, it.bbox)
            idxGraphs.insert(len(graphBoxes)+count, length2coor(box))
            count += 1
            # graphBoxes.append((n, box))
            # n += 1
    # print(idxGraphs.get_bounds())
    # print(idxGraphs.leaves())
    # exit(0)
    if len(graphBoxes) > 0:
        graphItems = list(idxGraphs.intersection(idxGraphs.get_bounds(), objects=True))
        graphBoxes = [(i.id, coor2length(num2int(i.bbox))) for i in graphItems]

    for j,b in graphBoxes:
        bEx = expand_box(b)
        wordBoxes = [it for it in wordBoxes if not is_within(it[1], bEx) and it[1][3] < it[1][2]*1.5]
    
    items = [(it, 0) for it in wordBoxes] + [(it, 1) for it in graphBoxes]
    idxItems = index.Index()
    for i,it in enumerate(items):
        idxItems.insert(it[0][0], length2coor(it[0][1]), obj=it[1])

    count = 0
    # flag = False
    while True:
        tempItems = []
        for it in items:
            itemInter = list(idxItems.intersection(length2coor(it[0][1]), objects=True))
            if len(itemInter) > 1:
                boxes = [num2int(i.bbox) for i in itemInter]
                box = fuse_box(boxes)
                for it1 in itemInter:
                    idxItems.delete(it1.id, it1.bbox)
                idxItems.insert(len(items)+count, length2coor(box), obj=1)
                count += 1
                itemExs = list(idxItems.intersection(idxItems.get_bounds(), objects=True))
                tempItems = [((i.id, coor2length(num2int(i.bbox))), i.object) for i in itemExs]
                break
        if len(tempItems) > 0:
            items = tempItems.copy()
        else:
            itemExs = list(idxItems.intersection(idxItems.get_bounds(), objects=True))
            items = [((i.id, coor2length(num2int(i.bbox))), i.object) for i in itemExs]
            break

    # Sort
    items.sort(key=take_key)

    g = 0
    for j,it in enumerate(items):
        c = length2coor(it[0][1])
        item = {}
        item["id"] = j
        item["location"] = coor2dic(c)
        if it[1] == 0:
            item["type"] = "text"
            words = bdResult["words_result"][it[0][0]]["words"]
            item["value"] = words
        elif it[1] == 1:
            item["type"] = "graph"
            itemPath = os.path.join(itemDir, str(idGraph+g)+'.png')
            cv2.imwrite(itemPath, imageRegion[c[1]:c[3], c[0]:c[2]])
            item["value"] = itemPath
            g += 1
        xiaoti["content"].append(item)
    return xiaoti


def deal_one_page(srcPath, desPath='', stanPath='', saveImage=True, charsOnly=False, rectification=False):
    """
    """
    global heightOfLine
    coor5 = []
    mistakes = 0
    bdResult = {}
    name = os.path.basename(srcPath)
    if not os.path.exists(srcPath):
        print("Image path not exists!")
        return None
    try:
        imgBgr = cv2.imread(srcPath)
        if rectification:
            imgBgr = rectify(imgBgr.copy())
        imgData = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2GRAY)
        
        # imgData = cv2.imread(srcPath, cv2.IMREAD_GRAYSCALE)
        # imgBgr = cv2.cvtColor(imgData, cv2.COLOR_GRAY2BGR)
    except Exception as imageError:
        print(imageError, 'Could not read the image file, skip...')

    # cv2.namedWindow("imgData", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("imgData", imgData)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    imgEliVer = eli_ver(imgData.copy())   #
    # cv2.namedWindow("imgEliVer", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("imgEliVer", imgEliVer)

    coor1 = divide_ver(imgEliVer)   #
    # print('coor1: {}'.format(coor1))

    if heightOfLine == 0:   #
        coor2 = divide_hor(imgEliVer, coor1)
        heightOfLines = [c[3]-c[1] for c in coor2]
        heightOfLine = median(heightOfLines)

    imgInv = imu.preprocess_bw_inv(imgData.copy())
    cv2.imwrite('imgInv.jpg', imgInv)
    coorCC = get_no_intersect_boxes('imgInv.jpg')
    imgEliCC = eli_large_cc(coorCC, imgData.copy(), heightOfLine)
    os.remove('imgInv.jpg')
    # cv2.namedWindow("imgEliCC", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("imgEliCC", imgEliCC)

    imgEli = eli_ver(imgEliCC) #1
    # cv2.namedWindow("imgEli", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("imgEli", imgEli)

    coor2 = divide_hor(imgEli, coor1)   #2
    # print('coor2: {}'.format(coor2))
    # img2 = mark_box(imgBgr, coor2, color=(0,255,0))
    # cv2.namedWindow("img2", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("img2", img2)

    coor3 = find_char(imgEli, coor2)  #3
    # print(np.array(coor3))
    # for l in coor3:
    #     img3 = mark_box(imgBgr, l, color=(0,255,0))
    # cv2.namedWindow("img3", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("img3", img3)

    if charsOnly:
        return coor3

    # print('SVM classifying...')
    coor4 = svm_classify(imgEli, coor3)    #4
    # print(*coor4, sep='\n')

    imgBw = imu.preprocess_bw(imgEli, boxSize=(4,4), morph=False) #re
    coor3Bw = find_char(imgBw, coor2)
    coor4Bw = svm_classify(imgBw, coor3Bw)
    coor5 = update_result(coor4, coor4Bw)

    try:
        # imgBd = preprocess_bw(imgEliCC)
        # print('BaiDu recognizing...')
        bdResult = bd_rec(imgEliCC, client, imgForm=srcPath[-4:], api='general')
    # except ConnectionResetError as e:
    #     print('Connection reset by peer, repeat BaiDu recognizion...')
    #     bdResult = bd_rec(imgEliCC, client, api='accurate')
    except Exception as e:
        print('Baidu recognition error, check your internet connection. exit...')
        exit(0)
    # print(bdResult)
    addResult = additional_questions(bdResult)
    # print(addResult)
    if addResult:
        coor5.append(([addResult], 2))
    # print(*coor4, sep='\n')

    if stanPath:
        stanResult = output_stan(coor5, os.path.basename(srcPath))
        mistakes = verification(stanResult, stanPath)
    
    # print(mistakes)

    # if saveImage:
    # for l in coor5:   #5
    #     # print(l)
    #     color = (255,255,0)
    #     if l[1] != 0:
    #         color = (255,0,0) if l[1] == 2 else (0,255,0)
    #     img5 = mark_box(imgBgr, l[0], color)
    # cv2.imwrite(desPath, img5)    #6

    # show_image(img5)
    # exit(0)
    # cv2.namedWindow("img5", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("img5", img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # imgEliVer2 = detect_vertical_line2(imgData)
    # coor2 = divide_hor(imgEliVer2, coor1)
    sa = detect_self_assessment(bdResult)
    if sa:
        # imgEliSa = imgData.copy()
        imgData[sa:, coor1[1][0]:coor1[1][1]+10] = 255
        # coor22 = divide_hor(imgEliSa, coor1)
    
    coor22 = divide_hor(imgData, coor1)
    region = integrate_lines(coor5, coor22, imgData.shape[1], imgData.shape[0])
    # print('region: {}'.format(region))
    # imgEliVerBgr = cv2.cvtColor(imgEliVer2, cv2.COLOR_GRAY2BGR)
    # imgRegion = mark_box(imgBgr, region, color=(0,255,0))
    # cv2.imwrite(desPath, imgRegion)

    # cv2.imwrite('imgInv.jpg', get_dilation(imgData.copy()))
    # coorCCDil = get_no_intersect_boxes('imgInv.jpg')
    # os.remove('imgInv.jpg')
    '''
    CCAndType = []
    wordBoxes = [w["location"] for w in bdResult["words_result"]]
    idxWords = index.Index()
    for j,b in enumerate(wordBoxes):
        idxWords.insert(j, (b["left"], b["top"], b["left"]+b["width"], b["top"]+b["height"]))

    idxRegion = index.Index()
    for j,r in enumerate(region):
        idxRegion.insert(j, r)

    idxCC = index.Index()
    for j,c in enumerate(coorCC):
        idxCC.insert(j, (c[0], c[1], c[0]+c[2], c[1]+c[3]))

    boxOfSpecialWord = get_match_location(bdResult, keyWordsConfig)
    hitsOfCC = []
    for j,s in enumerate(boxOfSpecialWord):
        b = (s[0], s[1], s[0]+s[2], s[1]+s[3])
        hitsOfRegion = list(idxRegion.intersection(b, objects=True))
        items = [item for item in hitsOfRegion if item.bbox[1] < s[1]+s[3]/2 < item.bbox[3]]
        for item in items:
            boxOfGraph = (item.bbox[0], b[3]+10, item.bbox[2]-item.bbox[0], item.bbox[3]-b[3]-20)
            boxOfGraph1 = (item.bbox[0], b[3]+10, item.bbox[2], item.bbox[3]-10)
            coorCC.append(boxOfGraph)
            hitsOfCC += list(idxCC.intersection(boxOfGraph1, objects=False))
    # print(hitsOfCC)
    coorCCCopy = coorCC.copy()
    coorCC = [coorCCCopy[i] for i in range(len(coorCCCopy)) if i not in hitsOfCC]

    boxRest = []
    for i,c in enumerate(coorCC.copy()):
        r = (c[0], c[1], c[0]+c[2], c[1]+c[3])
        hitsOfRegion = list(idxRegion.intersection(r))
        if len(hitsOfRegion) == 0:
            continue

        if c[2] > heightOfLine*3 or c[3] > heightOfLine*1.5:
            CCAndType.append((c, 1))
            # del(coorCC[i])
            continue
        elif c[2] < heightOfLine*0.5 and c[3] < heightOfLine*0.5:
            CCAndType.append((c, 0))
            continue
        
        hits = list(idxWords.intersection(r, objects=True))
        if hits:
            flag = False
            for item in hits:
                wordBox = (wordBoxes[item.id]["left"], wordBoxes[item.id]["top"], wordBoxes[item.id]["width"], wordBoxes[item.id]["height"])
                if is_within(c, wordBox):
                    CCAndType.append((c, 0))
                    flag = True
                    # del(coorCC[i])
                    continue
            if flag:
                continue
        boxRest.append(c)

    typesOfCC = tell_me_text_or_graph(imgBgr.copy(), imgData.copy(), boxRest)
    CCAndType += zip(boxRest, typesOfCC)
    resultFinal = get_items(region, imgData.copy(), bdResult, CCAndType, name)
    '''
    imgData1 = detect_vertical_line3(imgData.copy())
    resultFinal = get_items_v02(region, imgData1.copy(), coorCC, name)
    # coorRegion = []
    # coorGraph = []
    # coorText = []
    # resultFinalA = absolute_coor(resultFinal)
    # for x in resultFinalA["questions"]:
    #     coorRegion.append(dic2coor(x["location"]))
    #     coorGraph.extend([dic2coor(i["location"]) for i in x["content"] if i["type"]=="graph"])
    #     coorText.extend([dic2coor(i["location"]) for i in x["content"] if i["type"]=="text"])

    # imgRegion = mark_box(imgBgr, coorRegion, color=(0,255,0))
    # imgGraph = mark_box(imgRegion, coorGraph, color=(0,0,255))
    # imgText = mark_box(imgGraph, coorText, color=(255,255,0))

    if desPath:
        with open(desPath[:-4]+'.json', 'w') as f:
            json.dump(resultFinal, f)
        # if saveImage:
        #     cv2.imwrite(desPath, imgText)
    # cv2.namedWindow("imgData", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("imgData", imgData)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
   
    #return coor5, mistakes
    return resultFinal
    

def batch(srcDir1, desDir1, rectification=False):
    """Batch process."""
    global srcDir, desDir
    srcDir, desDir = srcDir1, desDir1
    init()
    imgPathList = os.listdir(srcDir)
    # imgPathList = ['120190703153929308.jpg']
    # imgPathList = ['math1-2917-1-2.png', 'math1-2017-1-1.png', 'math3-2017-6-1.png', 
    #                 'math2-2017-6-1.png', 'math3-2017-6-2.png', 'math2-2017-6-2.png']
    imgPathList.sort()
    valids = 0
    wrongList = []
    totalMistakes = 0
                    
    for n,i in enumerate(imgPathList):
        if not i.endswith(('.jpg', '.png')): 
            print('{} is not a jpg and png image file, skip...'.format(i))
            continue
        srcPath = os.path.join(srcDir, i)
        desPath = os.path.join(desDir, i)
        # stanPath = os.path.join(stanDir, i[:-4]+'.json')
        print('{}: {}'.format(n, srcPath))
        if not os.path.exists(srcPath):
            print('File path not exists! skip...')
            continue

        # coordinates, mistakes = deal_one_page(srcPath, desPath, stanPath='', saveImage=False)
        deal_one_page(srcPath, desPath, saveImage=False, charsOnly=False, rectification=rectification)
        # if mistakes:
        #     totalMistakes += mistakes
        #     wrongList.append((i, mistakes))

        # paper = Paper(name=i, image=imgData, coor=coordinates)
        # paper.generate()
        # paper.structure()
        # if not paper.valid:
        #     wrongList.append(i)
        # else:
        #     valids = valids + 1

        print('---------------------------------------------')
    # print('Height of line = {}'.format(heightOfLine))
    # print('marginButtom = {}'.format(marginButtom))
    # print('marginTop = {}'.format(marginTop))
    # print('Valids = {}'.format(valids))
    # print('Total mistakes = {}'.format(totalMistakes))
    # print('Mistakes list: {}'.format(wrongList))


def extract_head_of_row(srcPath, rectification):
    return deal_one_page(srcPath=srcPath, charsOnly=True, rectification=rectification)


def extract_questions(srcDir, desDir, rectification, config):
    global xmlPath
    xmlPath = config
    return batch(srcDir1=srcDir, desDir1=desDir, rectification=rectification)



# Initialisation starts --------------------------------------------

# srcDir = './img/hough_angle/biaozhundashijuan'    #
# srcDir = './img/sample'    #
# desDir = './res/hough_angle_result_0827/biaozhundashijuan'
# desDir = './res/sample0826'
# stanDir = './sta'
# coorDir = './coor/biaozhundashijuan'
# modelPath = '../svm/model/title_model06.m'
# modelPath = 'svm-model.pickle'
# xmlPath = '../configs/paper.xml'
keyWordsConfig = '../configs/key_words.txt'

# srcDir = ''
# desDir = ''
heightOfLine = 0
marginButtom = 0
marginTop = 0
client = ''
keyWords = ''

def init():
    global keyWords, marginTop, marginButtom, client

    if not os.path.isdir(srcDir):
        print('Direction error! exit...')
        exit(0)

    with open(keyWordsConfig,'r', encoding='UTF-8') as f:
        for i in f.readlines():
            keyWords = keyWords + str(i.strip())
            keyWords = keyWords + "|"
    keyWords = keyWords[:-1]

    DOMTree=parse(xmlPath)
    paperlist=DOMTree.documentElement
    papers=paperlist.getElementsByTagName('paper')
    for p in papers:
        # if p.hasAttribute('category'):
        #      # print ('category is ', book.getAttribute('category'))
        if p.getAttribute('category') == os.path.basename(srcDir):
            marginButtom = int(p.getElementsByTagName('marginButtom')[0].childNodes[0].data)
            marginTop = int(p.getElementsByTagName('marginTop')[0].childNodes[0].data)
            break

    client = bd_access()


# Initialisation ends ---------------------------------------------



# Main starts --------------------------------------------
if __name__ == '__main__':
    #import warnings
    #warnings.filterwarnings("ignore")
    startT = time.process_time()
    batch()
    endT = time.process_time()
    print('Total time: %s'%(endT - startT))
# Main ends --------------------------------------------



