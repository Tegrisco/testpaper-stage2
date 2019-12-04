import json
import numpy as np
import cv2
import subprocess
import os
import shutil
from rtree import index
import sys
sys.path.append('../packages/')
from blob_kit.base import label_exe
'''
找到target对应的box，和这个box左右两边的box 
'''
out_dir = '../packages/blob_kit/tmp'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def preprocess_bw_inv(gray, smooth=True, thresh=200, morph=True):

    if smooth:
        gray = cv2.boxFilter(gray, -1, (3, 3))

    ret, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    if morph:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, element)

    # cv2.imshow('image1', bw2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # bw2[bw2 == 255] = 1
    return bw

def get_blob_contours(black_image_path):
    """
    :param black_image_path: 二值化后图像的路径
    :return: contours: 轮廓列表，OpenCV 格式。(N,1,2) 维,N 为边界点的个数
    """
    base, file = os.path.split(black_image_path)
    cmd = '{0} {1} {2}'.format(label_exe, black_image_path, os.path.join(out_dir, 'label.png'))
    subprocess.run(cmd, shell=True)
    if os.path.exists(os.path.join(out_dir, 'blob.json')):
        os.remove(os.path.join(out_dir, 'blob.json'))
    shutil.move('blob.json', out_dir)
    os.remove('blob.plot')
    contours = blob_parse(os.path.join(out_dir, 'blob.json'))
    return contours


def blob_parse(bfile):
    """

    :param bfile: blob json 文件路径
    :return: 轮廓列表，OpenCV 格式。(N,1,2) 维,N 为边界点的个数
    """

    fp = open(bfile, 'r')
    y = json.load(fp)
    fp.close()
    cnts = []
    for i in range(len(y['blobs'])):
        # print(y['blobs'][i]['external'])
        x = np.array(y['blobs'][i]['external']).reshape((-1,1,2))
        cnts.append(x)
    return cnts


def reduce_contours(cnts, width_thres=2, height_thres=2):
    """
    应更扩展
    :param cnts:轮廓列表，OpenCV 格式。(N,1,2) 维,N 为边界点的个数
    :param width_thres:阈值
    :param height_thres:阈值
    :return:coutours:轮廓列表，OpenCV 格式。(N,1,2) 维,N 为边界点的个数
    """
    contours = []
    boxes = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > width_thres and h > height_thres:
            contours.append(cnt)
            boxes.append((x, y, w, h))
    return contours, boxes


def is_within(box1, box2):
    return box1[0] >= box2[0] and box1[1] >= box2[1] and (box1[0]+box1[2] <= box2[0]+box2[2]) and (box1[1]+box1[3] <= box2[1]+box2[3])

def reduce_overlapping_boxes(boxes):
    """

    :param boxes:所有的box
    :return:final_boxes是去除大box中的小的box后剩下的box
    """
    final_boxes = []
    for i in range(0, len(boxes)):
        flag = False
        for a in range(0, len(boxes)):
            if i == a:
                continue
            else:
                # if boxes[a][0] <= boxes[i][0] <= boxes[a][0] + boxes[a][2]-1 and boxes[a][1] <= boxes[i][1] <= boxes[a][1] + boxes[a][-1]-1:
                if is_within(boxes[i], boxes[a]):

                    # print(boxes[i], boxes[a])
                    flag = True
                    # continue
                    break
                # else:
        if not flag:
            final_boxes.append(boxes[i])

    return final_boxes

# final_boxes[(x,y,w,h),...]


def get_final_box(black_image_path):
    """
    请注意：black_image_path是二值化之后的图片路径，此函数可以输入二值化后图片路径得到final_boxes
    :return:
    """
    cnts = get_blob_contours(black_image_path)
    cnts, boxes = reduce_contours(cnts)
    final_boxes = reduce_overlapping_boxes(boxes)
    return final_boxes


def get_no_intersect_boxes(bw_image_path):
    """

    :param bw_image_path: 二值化图像路径，非零表示前景
    :return: 返回 (x,y,w,h) 格式的 box 列表
    """
    cnts = get_blob_contours(bw_image_path)
    cnts, boxes = reduce_contours(cnts)

    # hp.dump2json(boxes, 0)
    # boxes = hp.loadjson(0)

    rtboxes = [(b[0],b[1],b[0]+b[2]-1,b[1]+b[3]-1) for b in boxes]


    # for box in rtboxes:
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
    # imu.imwrite_(img, 3)

    idx = index.Index()

    # for i in range(len(rtboxes)):
    #     box = rtboxes[i]
    #     ids = list(idx.intersection(box))
    #     if ids:
    #         leftv = [box[0]]
    #         bottomv = [box[1]]
    #         rightv = [box[2]]
    #         upv = [box[3]]
    #
    #         for id in ids:
    #             leftv.append(rtboxes[id][0])
    #             bottomv.append((rtboxes[id][1]))
    #             rightv.append((rtboxes[id][2]))
    #             upv.append(rtboxes[id][3])
    #             idx.delete(id, rtboxes[id])
    #
    #             rtboxes[id] = None
    #         left = min(leftv)
    #         bottom = min(bottomv)
    #         right = max(rightv)
    #         up = max(upv)
    #         rtboxes[i] = (left, bottom, right, up)
    #         idx.insert(i, rtboxes[i])
    #     else:
    #         idx.insert(i, box)

    i = 0
    while i < len(rtboxes):
        box = rtboxes[i]
        ids = list(idx.intersection(box))
        if ids:
            leftv = [box[0]]
            bottomv = [box[1]]
            rightv = [box[2]]
            upv = [box[3]]

            for id in ids:
                leftv.append(rtboxes[id][0])
                bottomv.append((rtboxes[id][1]))
                rightv.append((rtboxes[id][2]))
                upv.append(rtboxes[id][3])
                idx.delete(id, rtboxes[id])

                rtboxes[id] = None
            left = min(leftv)
            bottom = min(bottomv)
            right = max(rightv)
            up = max(upv)
            rtboxes[i] = (left, bottom, right, up)
            # idx.insert(i, rtboxes[i])
        else:
            idx.insert(i, box)
            i = i+1


    result = [(box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1) for box in rtboxes if box is not None]
    # print(idx.count(idx.get_bounds()))
    return result



def box_iou(box1, box2):
    """
    计算两个box的IOU
    :param box1:一个box
    :param box2:另一个box
    :return:iou是指box1和box2的重合程度（交集面积除以并集面积）
    """
    long1 = {x for x in range(box1[0], box1[0]+box1[2])}
    long2 = {x for x in range(box2[0], box2[0]+box2[2])}
    wide1 = {x for x in range(box1[1], box1[1]+box1[3])}
    wide2 = {x for x in range(box2[1], box2[1] + box2[3])}

    if (len(long1 & long2) == 0 and len(wide1 & wide2) == 0) or len(long1 & long2) == 1 or len(wide1 & wide2) == 1:
        iou = 0
    else:
        s1 = len(long1 & long2) * len(wide1 & wide2)
        s2 = len(long1) * len(wide1) + len(long2) * len(wide2) - s1
        iou = s1 / s2
    return iou


def get_box_covered(boxes, target, iou_thres=0.5):
    """
    在 boxes 中找到 target 覆盖最大的 box。若 IOU 都小于 iou_thres，返回None
    :param boxes:所有的box
    :param target:规定的范围
    :param iou_thres:阈值
    :return:max_box是指target中与target重合程度最大的box
    """

    max_box = None
    max_iou = 0.0
    for box in boxes:
        iou = box_iou(box, target)
        if iou > iou_thres and iou > max_iou:
            max_iou = iou
            max_box = box
    return max_box


def get_box_left(boxes, target):
    """
    在 boxes 中找到 target 左边的 box。
    先 get_box_covered，若 IOU 都小于 iou_thres，返回 None
    若找不到左边的box，返回 None
    :param boxes:所有的box
    :param target:规定范围
    :param iou_thres:阈值
    :return:left_box为target中对应的box左边的box
    """
    left_box = None
    min_dic = 50
    max_box = get_box_covered(boxes, target, iou_thres=0.1)
    if max_box == None:
        return None
    left_target = (max_box[0] - 100, max_box[1], 100, 50)
    for box in boxes:
        iou = box_iou(box, left_target)
        if iou > 0:
            dic = box[0] + box[2] - max_box[0]
            if dic < min_dic:
                min_dic = dic
                left_box = box
    return left_box


def get_box_right(boxes, target):
    """
    在 boxes 中找到 target 右边的 box。
    先 get_box_covered，若 IOU 都小于 iou_thres，返回 None
    若找不到右边的box，返回 None
    :param boxes:所有的box
    :param target:规定的范围
    :param iou_thres:阈值
    :return:right_box指target中对应的box右边的box
    """
    min_dic = 30
    right_box = None
    max_box = get_box_covered(boxes, target, iou_thres=0.0)
    right_target = (max_box[0] + max_box[2], max_box[1], 100, 50)
    for box in boxes:
        iou = box_iou(box, right_target)
        if iou > 0:
            dic = box[0] - max_box[0] - max_box[2]
            if dic < min_dic:
                min_dic = dic
                right_box = box
    return right_box


def seek_right_box(white_image_path, black_image_path, locationOfChar):
    """
    找到所有的target对应的left_box,right_box
    :param  white_image_path: 白底图片路径
    :param black_image_path: 黑底图片储存路径
    :param locationOfChar: 所有的你规定的target范围
    :return: img上暂时只画了target的框，如有需要可以增加框，right_boxes,left_boxes储存所有的左边和右边box为一个列表
    """
    gray = cv2.imread(white_image_path, 0)
    right_boxes = []
    left_boxes = []
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(locationOfChar)):
        target = [locationOfChar[i][0]['left'], locationOfChar[i][0]['top'], locationOfChar[i][0]['width'], locationOfChar[i][0]['height']]
        img, right_box = test_get_box_right(img, target, black_image_path)
        left_box = test_get_box_left(img, target, black_image_path)
        right_boxes.append(right_box)
        left_boxes.append(left_box)
        cv2.rectangle(img, (target[0], target[1]), (target[0] + target[2], target[1] + target[3]), (0, 0, 250), 2)
    return img, right_boxes, left_boxes


def test_get_box_left(img, target, black_image_path):
    cnts = get_blob_contours(black_image_path)
    cnts, boxes = reduce_contours(cnts)
    final_boxes = reduce_overlapping_boxes(boxes)
    left_box = get_box_left(boxes, target)
    return left_box


def test_get_box_right(img, target, c):
    cnts = get_blob_contours(c)
    cnts, boxes = reduce_contours(cnts)
    final_boxes = reduce_overlapping_boxes(boxes)
    right_box = get_box_right(boxes, target)
    return img, right_box



if __name__ == '__main__':

    srcdir = os.path.join(data_dir, 'waibu/biaozhundashijuan')
    # srcdir = '../img'

    out_base = '../result/boxes'
    outdir = os.path.join(out_base, 'biaozhundashijuan')
    # outdir = './tmp'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # papers = os.listdir(srcdir)
    papers = ['120190703153331060.jpg']

    for paper in papers:
        if paper.endswith('.jpg'):
            imgfile = os.path.join(srcdir, paper)
            outfile = os.path.join(outdir, paper)

            print(imgfile)

            img = cv2.imread(imgfile)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw = preprocess_bw_inv(gray)
            cv2.imwrite('./tmp/bw.jpg', bw)
            boxes = get_no_intersect_boxes('./tmp/bw.jpg')

            for box in boxes:
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2] - 1, box[1] + box[3] - 1), (0, 0, 255), 1)

            cv2.imwrite(outfile, img)








