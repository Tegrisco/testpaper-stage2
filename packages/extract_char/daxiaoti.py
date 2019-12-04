import json
import numpy as np
import cv2
import os
import time
from rtree import index
import sys

sys.path.append('../packages/')
# from src.base import data_dir
import extract_char.helpers as hp
import imutils as imu
import bl_div_kit as bl
from blob_kit.blob_analysis import box_iou


box_whto2p = lambda boxwh: (boxwh[0],boxwh[1], boxwh[0]+boxwh[2]-1, boxwh[1]+boxwh[3]-1)
box_2ptowh = lambda box2p: (box2p[0],box2p[1], box2p[2]-box2p[0]+1, box2p[3]-box2p[1]+1)

def hori_portion(box_ref, box2):
    """
    水平重合部分占 box_ref 宽度的比例
    box 为 x,y,w,h格式
    :param box_ref:
    :param box2:
    :return:
    """
    left = max(box_ref[0], box2[0])
    right = min(box_ref[0] + box_ref[2], box2[0] + box2[2])

    if right > left:
        return (right - left) / box_ref[2]
    else:
        return 0.0

def save_char_img():

    # print(box_2ptowh((3,4,7,8)))
    # print(box_whto2p(box_2ptowh((3,4,7,8))))
    # exit(0)

    books = ['liangdiangeili', 'biaozhundashijuan', 'jiangsumijuan']
    # books = ['jiangsumijuan']

    for book in books:
        srcdir = os.path.join(data_dir, 'waibu/' + book)

        baidu_dir = os.path.join('../dataset/baidu', book)
        CC_dir = os.path.join('../dataset/CC', book)
        f3_dir = os.path.join('../dataset/f3', book)

        out_base = '../result/daxiaoti'
        outdir = os.path.join(out_base, book)

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        char_dir = '../dataset/char'
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)

        papers = os.listdir(srcdir)
        # papers = ['120190703153513729.jpg']

        for paper in papers:
            if paper.endswith('.jpg'):

                print(os.path.join(srcdir, paper))
                img = cv2.imread(os.path.join(srcdir, paper))

                ocr = hp.loadjson(paper, dir=baidu_dir)
                CC = hp.loadjson(paper, dir=CC_dir)
                f3 = hp.loadjson(paper, dir=f3_dir)

                # for words in ocr['contentDic']["words_result"]:
                #     # cv2.rectangle(img, (words['location']['left'], words['location']['top']), (
                #     # words['location']['left'] + words['location']['width'], words['location']['top'] + words['location']['height']),
                #     #               (255, 0, 0), 2)
                #     for c in words['chars']:
                #         cv2.rectangle(img, (c['location']['left'], c['location']['top']), (
                #         c['location']['left'] + c['location']['width'], c['location']['top'] + c['location']['height']),
                #                       (0, 0, 255))

                # for c in ocr['title']:
                #     cv2.rectangle(img, (c[0]['left'], c[0]['top']),
                #                   (c[0]['left'] + c[0]['width'], c[0]['top'] + c[0]['height']), (255, 255, 0))
                #
                #
                # for box in CC:
                #     cv2.rectangle(img, (box[0],box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0))

                idx = index.Index()
                rtboxes = [(b[0], b[1], b[0] + b[2] - 1, b[1] + b[3] - 1) for b in CC]
                for i, box in enumerate(rtboxes):
                    idx.insert(i, box)

                char = '八'
                if char == '.':
                    curr_save_dir = hp.mkdir(os.path.join(char_dir, 'dot'))
                else:
                    curr_save_dir = hp.mkdir(os.path.join(char_dir, char))

                char_loc = []
                for words in ocr['contentDic']["words_result"]:
                    # cv2.rectangle(img, (words['location']['left'], words['location']['top']), (
                    # words['location']['left'] + words['location']['width'], words['location']['top'] + words['location']['height']),
                    #               (255, 0, 0), 2)
                    # for c in words['chars']:
                    #     if c['char'] == '一':
                    #         cv2.rectangle(img, (c['location']['left'], c['location']['top']), (
                    #             c['location']['left'] + c['location']['width'],
                    #             c['location']['top'] + c['location']['height']),
                    #                       (0, 0, 255))
                    locs = [(c['location']['left'], c['location']['top'],
                             c['location']['left'] + c['location']['width'] - 1,
                             c['location']['top'] + c['location']['height'] - 1) for c in words['chars'] if
                            c['char'] == char]
                    char_loc.extend(locs)

                k = 0
                img_draw = img.copy()

                for loc in char_loc:
                    cv2.rectangle(img_draw, (loc[0], loc[1]), (loc[2], loc[3]), (255, 0, 0), 2)
                    cc = list(idx.intersection(loc))
                    # cc = list(idx.nearest(loc))
                    if cc:
                        ious = [hori_portion(box_2ptowh(rtboxes[i]), box_2ptowh(loc)) for i in cc]
                        cc_proposal = [rtboxes[cc[i]]  for i,iou in enumerate(ious) if iou>0.2]
                        if not cc_proposal:
                            continue
                        left = min([box[0] for box in cc_proposal])
                        top = min([box[1] for box in cc_proposal])
                        right = max([box[2] for box in cc_proposal])
                        bottom = max([box[3] for box in cc_proposal])

                        loc2 = (left, top, right, bottom)
                        if (loc2[2]-loc2[0])<70 and (loc2[3]-loc2[1])<70:
                        # if (loc2[2] - loc2[0]) < 20 and (loc2[3] - loc2[1]) < 20: # for dot
                            cv2.rectangle(img_draw, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)

                            cv2.imwrite(os.path.join(curr_save_dir, book + paper + '_' + str(k) + '.jpg'),
                                        img[loc2[1]:loc2[3] + 1, loc2[0]:loc2[2] + 1])
                            k = k + 1

                # char = '.'
                # if char == '.':
                #     curr_save_dir = hp.mkdir(os.path.join(char_dir, 'dot'))
                # else:
                #     curr_save_dir = hp.mkdir(os.path.join(char_dir, char))
                #
                # char_loc = []
                # for words in ocr['contentDic']["words_result"]:
                #     # cv2.rectangle(img, (words['location']['left'], words['location']['top']), (
                #     # words['location']['left'] + words['location']['width'], words['location']['top'] + words['location']['height']),
                #     #               (255, 0, 0), 2)
                #     # for c in words['chars']:
                #     #     if c['char'] == '一':
                #     #         cv2.rectangle(img, (c['location']['left'], c['location']['top']), (
                #     #             c['location']['left'] + c['location']['width'],
                #     #             c['location']['top'] + c['location']['height']),
                #     #                       (0, 0, 255))
                #     locs = [(c['location']['left'], c['location']['top'],
                #              c['location']['left'] + c['location']['width'] - 1,
                #              c['location']['top'] + c['location']['height'] - 1) for c in words['chars'] if
                #             c['char'] == char]
                #     char_loc.extend(locs)
                #
                # k = 0
                # img_draw = img.copy()
                #
                # for loc in char_loc:
                #     cv2.rectangle(img_draw, (loc[0], loc[1]), (loc[2], loc[3]), (255, 0, 0), 2)
                #     cc = list(idx.intersection(loc))
                #     # cc = list(idx.nearest(loc))
                #     if cc:
                #         istar = 0
                #         if len(cc) > 1:
                #             # ious = [box_iou(box_2ptowh(loc), box_2ptowh(rtboxes[i])) for i in cc]
                #             ious = [hori_portion(box_2ptowh(loc), box_2ptowh(rtboxes[i])) for i in cc]
                #             istar = np.argmax(ious)
                #
                #         loc2 = rtboxes[cc[istar]]
                #         if (loc2[2]-loc2[0])<70 and (loc2[3]-loc2[1])<70:
                #         # if (loc2[2] - loc2[0]) < 20 and (loc2[3] - loc2[1]) < 20: # for dot
                #             cv2.rectangle(img_draw, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)
                #
                #             cv2.imwrite(os.path.join(curr_save_dir, book + paper + '_' + str(k) + '.jpg'),
                #                         img[loc2[1]:loc2[3] + 1, loc2[0]:loc2[2] + 1])
                #             k = k + 1

                cv2.imwrite(os.path.join(outdir, paper), img_draw)


import random
def save_other_char_img():


    books = ['liangdiangeili', 'biaozhundashijuan', 'jiangsumijuan']
    # books = ['jiangsumijuan']

    for book in books:
        srcdir = os.path.join(data_dir, 'waibu/' + book)

        baidu_dir = os.path.join('../dataset/baidu', book)
        CC_dir = os.path.join('../dataset/CC', book)
        f3_dir = os.path.join('../dataset/f3', book)

        out_base = '../result/daxiaoti'
        outdir = os.path.join(out_base, book)

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        char_dir = '../dataset/char'
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)

        papers = os.listdir(srcdir)
        # papers = ['120190703153513729.jpg']

        for paper in papers:
            if paper.endswith('.jpg'):

                print(os.path.join(srcdir, paper))
                img = cv2.imread(os.path.join(srcdir, paper))

                ocr = hp.loadjson(paper, dir=baidu_dir)
                CC = hp.loadjson(paper, dir=CC_dir)

                idx = index.Index()
                rtboxes = [(b[0], b[1], b[0] + b[2] - 1, b[1] + b[3] - 1) for b in CC]
                for i, box in enumerate(rtboxes):
                    idx.insert(i, box)

                char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '一', '二', '三', '四', '五', '六', '七', '八', '九', '、', '.', '十','E']
                curr_save_dir = hp.mkdir(os.path.join(char_dir, 'other'))

                char_loc = []
                for words in ocr['contentDic']["words_result"]:
                    locs = [(c['location']['left'], c['location']['top'],
                             c['location']['left'] + c['location']['width'] - 1,
                             c['location']['top'] + c['location']['height'] - 1) for c in words['chars'] if
                            c['char'] not in char_list]
                    char_loc.extend(locs)

                # sample char char_loc
                char_each_paper = 100
                char_loc = random.sample(char_loc, char_each_paper)

                k = 0
                img_draw = img.copy()

                for loc in char_loc:
                    cv2.rectangle(img_draw, (loc[0], loc[1]), (loc[2], loc[3]), (255, 0, 0), 2)
                    cc = list(idx.intersection(loc))
                    # cc = list(idx.nearest(loc))
                    if cc:
                        ious = [hori_portion(box_2ptowh(rtboxes[i]), box_2ptowh(loc)) for i in cc]
                        cc_proposal = [rtboxes[cc[i]]  for i,iou in enumerate(ious) if iou>0.2]
                        if not cc_proposal:
                            continue
                        left = min([box[0] for box in cc_proposal])
                        top = min([box[1] for box in cc_proposal])
                        right = max([box[2] for box in cc_proposal])
                        bottom = max([box[3] for box in cc_proposal])

                        loc2 = (left, top, right, bottom)
                        if (loc2[2]-loc2[0])<70 and (loc2[3]-loc2[1])<70:
                        # if (loc2[2] - loc2[0]) < 20 and (loc2[3] - loc2[1]) < 20: # for dot
                            cv2.rectangle(img_draw, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)

                            cv2.imwrite(os.path.join(curr_save_dir, book + paper + '_' + str(k) + '.jpg'),
                                        img[loc2[1]:loc2[3] + 1, loc2[0]:loc2[2] + 1])
                            k = k + 1



def save_char_img():


    books = ['liangdiangeili', 'biaozhundashijuan', 'jiangsumijuan']
    # books = ['jiangsumijuan']

    for book in books:
        srcdir = os.path.join(data_dir, book)

        baidu_dir = os.path.join('../res/baidu', book)
        CC_dir = os.path.join('../res/CC', book)
        f3_dir = os.path.join('../res/f3', book)

        out_base = hp.mkdir('../res/result/char')


        papers = os.listdir(srcdir)
        # papers = ['120190703153513729.jpg']

        k = 0

        for paper in papers:
            if paper.endswith('.jpg'):

                print(os.path.join(srcdir, paper))
                img = cv2.imread(os.path.join(srcdir, paper))

                ocr = hp.loadjson(paper, dir=baidu_dir)
                CC = hp.loadjson(paper, dir=CC_dir)

                idx = index.Index()
                rtboxes = [(b[0], b[1], b[0] + b[2] - 1, b[1] + b[3] - 1) for b in CC]
                for i, box in enumerate(rtboxes):
                    idx.insert(i, box)




                for words in ocr['contentDic']["words_result"]:
                    for c in words['chars']:
                        char = c['char']
                        loc = (c['location']['left'], c['location']['top'],
                             c['location']['left'] + c['location']['width'] - 1,
                             c['location']['top'] + c['location']['height'] - 1)

                        cc = list(idx.intersection(loc))
                        # cc = list(idx.nearest(loc))
                        if cc:
                            ious = [hori_portion(box_2ptowh(rtboxes[i]), box_2ptowh(loc)) for i in cc]
                            cc_proposal = [rtboxes[cc[i]] for i, iou in enumerate(ious) if iou > 0.2]
                            if not cc_proposal:
                                continue
                            left = min([box[0] for box in cc_proposal])
                            top = min([box[1] for box in cc_proposal])
                            right = max([box[2] for box in cc_proposal])
                            bottom = max([box[3] for box in cc_proposal])

                            loc2 = (left, top, right, bottom)
                            if (loc2[2] - loc2[0]) < 80 and (loc2[3] - loc2[1]) < 80:
                                # if (loc2[2] - loc2[0]) < 20 and (loc2[3] - loc2[1]) < 20: # for dot
                                # cv2.rectangle(img_draw, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)

                                if char=='.':
                                    char = 'dot'

                                curr_save_dir = hp.mkdir(os.path.join(out_base, char))

                                cv2.imwrite(os.path.join(curr_save_dir, book + paper + '_' + str(k) + '.jpg'),
                                            img[loc2[1]:loc2[3] + 1, loc2[0]:loc2[2] + 1])
                                k = k + 1



# import src.sklearn_svm as svm
#
# categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '一', '二', '三', '四', '五', '六', '七', '八', '九', '、', '.',
#               '十', 'E']
#
# def batch(srcDir, f3Dir, desDir):
#     """Batch process.
#
#     Args:
#         srcDir: Source images direction.
#         desDir: Destination images direction.
#
#     Returns:
#         None
#     """
#     classifier = svm.classifier()
#     freetyper = imu.freetyper()
#
#     if os.path.isdir(srcDir):
#         imgPathList = os.listdir(srcDir)
#         # imgPathList = ['120190703153354069.jpg']    #修改图片名
#         imgPathList.sort()
#
#         for n, i in enumerate(imgPathList):
#             if not i.endswith('.jpg'): continue
#             srcPath = os.path.join(srcDir, i)
#             desPath = os.path.join(desDir, i)
#
#             coor3 = hp.loadjson(i, dir=f3Dir)
#             # print(np.array(coor3))
#
#             try:
#                 gray = cv2.imread(srcPath, cv2.IMREAD_GRAYSCALE)
#                 imgBgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#             except Exception as imageError:
#                 print(imageError, 'Could not read the image file, skip...')
#                 continue
#
#             for m, l in enumerate(coor3):
#                 print('{} {}'.format(m, l))
#                 img3 = bl.mark_box(imgBgr, l, color=(0, 255, 0))
#
#             for line in coor3:
#                 for box in line:
#                     x1, y1, x2, y2 = box
#                     # label = classifier.predict_proba(gray[y1:y2, x1:x2], var_thres=0.03)
#                     label = classifier.predict(gray[y1:y2, x1:x2])
#
#                     # cv2.putText(img3, categories[label[0]], (x1, y1), font, 2, (0,0,255), 2)
#                     freetyper.putText(img3, categories[label[0]], (x1, y1 - 40))
#
#             cv2.imwrite(desPath, img3)
#
#
# def box_big_strech_intersect_combine(rtboxes, thresh=90, hstrech=30, vstrech=10):
#     """
#     相交的box合在一起
#     :param rtboxes:  (x1,y1,x2,y2) 格式
#     :return:   (x1,y1,x2,y2) 格式
#     """
#     idx = index.Index()
#     for i, box in enumerate(rtboxes):
#         idx.insert(i, box)
#
#     i = 0
#     while i < len(rtboxes):
#         box = rtboxes[i]
#
#         if box is None:
#             i=i+1
#             continue
#
#         # if box[2] < 200:
#         #     i = i + 1
#         #     continue
#
#         # if (box[3]-box[1]+1>60) and ((box[2]-box[0]+1>thresh) and (box[3]-box[1]+1>thresh)):
#         if  ((box[2]-box[0]+1 > 200) and (box[3]-box[1]+1 > 6)) or \
#                 ((box[2] - box[0] + 1 > thresh) and (box[3] - box[1] + 1 > thresh)):
#                 query_box = (box[0]-hstrech, box[1]-vstrech, box[2]+hstrech, box[3]+vstrech)
#         else:
#             i=i+1
#             continue
#
#
#         ids = list(idx.intersection(query_box))
#         if len(ids)>1:
#             leftv = []
#             bottomv = []
#             rightv = []
#             upv = []
#             for id in ids:
#                 leftv.append(rtboxes[id][0])
#                 bottomv.append((rtboxes[id][1]))
#                 rightv.append((rtboxes[id][2]))
#                 upv.append(rtboxes[id][3])
#
#                 idx.delete(id, rtboxes[id])
#                 rtboxes[id] = None
#             left = min(leftv)
#             bottom = min(bottomv)
#             right = max(rightv)
#             up = max(upv)
#             rtboxes[i] = (left, bottom, right, up)
#             idx.insert(i, rtboxes[i])
#         else:
#             # idx.insert(i, box)
#             i = i + 1
#
#     result = [box for box in rtboxes if box is not None]
#     return result
#
#
# def extract_graph():
#     books = ['biaozhundashijuan', 'jiangsumijuan', 'liangdiangeili']
#     # books = ['biaozhundashijuan']
#
#     for book in books:
#         startT = time.process_time()
#
#         srcdir = os.path.join(data_dir, 'waibu/' + book)
#         baidu_dir = os.path.join('../dataset/baidu', book)
#         CC_dir = os.path.join('../dataset/CC', book)
#
#
#         out_base = hp.mkdir('../result/graph')
#
#         outdir = hp.mkdir(os.path.join(out_base, book))
#
#         papers = os.listdir(srcdir)
#         # papers = ['120190703153331060.jpg']
#
#         for paper in papers:
#             if paper.endswith('.jpg'):
#
#                 print(os.path.join(srcdir, paper))
#                 img = cv2.imread(os.path.join(srcdir, paper))
#
#                 CC = hp.loadjson(paper, dir=CC_dir)
#                 ocr = hp.loadjson(paper, dir=baidu_dir)
#
#                 # # 足够大，才左右延伸
#                 # # 效果不好！
#                 # rtboxes = [(b[0], b[1], b[0] + b[2] - 1, b[1] + b[3] - 1) for b in CC]
#                 # rtboxes2 = box_big_strech_intersect_combine(rtboxes)
#                 #
#                 # for loc2 in rtboxes2:
#                 #     cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (255, 0, 0), 2)
#
#                 for box in CC:
#
#                         loc2 = box_whto2p(box)
#                         cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 255, 0), 2)
#
#
#                 #
#                 #
#                 #
#                 # hext = 15
#                 # CC2p = [box_whto2p(box) for box in CC]
#                 # CCext = [(box[0]-hext, box[1], box[2]+hext, box[3]) for box in CC2p]
#                 # CCext = blob.box_intersect_combine(CCext)
#                 #
#                 # CCext_tree = blob.Boxes(CCext)
#                 #
#                 #
#                 #
#                 # # # 建立 words result 的框列表
#                 # # words_result_loc = [(words['location']['left'], words['location']['top'],
#                 # #              words['location']['left'] + words['location']['width'] - 1,
#                 # #              words['location']['top'] + words['location']['height'] - 1) for words in ocr['contentDic']["words_result"]]
#                 # # chars_loc = []
#                 # # for words in ocr['contentDic']["words_result"]:
#                 # #     locs = [(c['location']['left'], c['location']['top'],
#                 # #              c['location']['left'] + c['location']['width'] - 1,
#                 # #              c['location']['top'] + c['location']['height'] - 1) for c in words['chars']]
#                 # #     chars_loc.extend(locs)
#                 # #
#                 # #
#                 # # # 建立 char 的框列表
#                 # #
#                 # # # 每一个cc 如果较小，且包含于 words result 里面，则删除
#                 # # words_boxes = blob.Boxes(words_result_loc)
#                 # # chars_boxes = blob.Boxes(chars_loc)
#                 # #
#                 # #
#                 # # # 所有 words result ,如果 iou 和 大CC 较大，则删除
#                 #
#                 # ratio = 0.8
#                 # lh = 60
#                 # vspan = int(lh*ratio)
#                 # for box in CC:
#                 #     if box[2]>80 or box[3]>80:
#                 #         loc2 = box_whto2p(box)
#                 #         cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 255, 0), 2)
#                 #
#                 #         l, u, r, b = loc2
#                 #         w = r-l
#                 #         hspan = min(int(0.8*w), 100)
#                 #         down_big_box = (l-hspan, b+1, r+hspan, b+vspan)
#                 #         loc2 = down_big_box
#                 #         cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)
#                 #
#                 #     else:
#                 #
#                 #         pass
#                 #
#                 # for loc2 in CCext:
#                 #     cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (255, 0, 0), 2)
#                 #
#                 #
#                 #
#                 #         # words_intersectoin = words_boxes.intersectioin(box_whto2p(box))
#                 #         # ious = [hori_portion(box, box_2ptowh(w)) for w in words_intersectoin]
#                 #         # if ious and max(ious) > 0.4:
#                 #         #     continue
#                 #         #
#                 #         # loc2 = box_whto2p(box)
#                 #         # cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (255, 0, 0), 2)
#                 #
#                 # # for words in ocr['contentDic']["words_result"]:
#                 # #
#                 # #     # locs = [(c['location']['left'], c['location']['top'],
#                 # #     #          c['location']['left'] + c['location']['width'] - 1,
#                 # #     #          c['location']['top'] + c['location']['height'] - 1) for c in words['chars']]
#                 # #     # for loc2 in locs:
#                 # #     #     cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (255, 0, 0), 2)
#                 # #
#                 # #     loc2 = (words['location']['left'], words['location']['top'],
#                 # #              words['location']['left'] + words['location']['width'] - 1,
#                 # #              words['location']['top'] + words['location']['height'] - 1)
#                 # #     cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)
#
#                 cv2.imwrite(os.path.join(outdir, paper), img)
#
#
#         endT = time.process_time()
#         print('Total time: %s' % (endT - startT))


# from src.base import data_dir

data_dir = '../img'

# Main starts --------------------------------------------
if __name__ == '__main__':

    # save_other_char_img()
    # exit(0)

    save_char_img()
    exit(0)

    # extract_graph()
    # exit(0)



    # books = ['biaozhundashijuan', 'jiangsumijuan', 'liangdiangeili']
    #
    # for book in books:
    #     startT = time.process_time()
    #
    #     srcdir = os.path.join(data_dir, 'waibu/' + book)
    #
    #     f3dir = os.path.join(data_dir, 'f3/' + book)
    #
    #     out_base = '../result/f3'
    #     if not os.path.exists(out_base):
    #         os.mkdir(out_base)
    #     outdir = os.path.join(out_base, book)
    #
    #     if not os.path.exists(outdir):
    #         os.mkdir(outdir)
    #
    #     batch(srcdir, f3dir, outdir)
    #
    #     endT = time.process_time()
    #     print('Total time: %s' % (endT - startT))


