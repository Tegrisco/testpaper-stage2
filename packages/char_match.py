# coding = UTF-8

"""
字符匹配
"""

import os
import cv2
import json
import time
import numpy as np

import sys
sys.path.append('../packages/')
from word_match.tools import *
import imutils as imu
from blob_kit.blob_analysis import get_no_intersect_boxes as get_cc

now = time.strftime('%Y%m%d%H%M%S')

class CharLibOcr():
    """基于字库匹配的ocr工具"""
    def __init__(self, char_lib_dir, dic_path, image):
        char_dic = load_json(dic_path)
        chars_list = os.listdir(char_lib_dir)
        self.image = image
        self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.char_lib = {}
        for char_name in chars_list:
            char_id = os.path.splitext(char_name)[0]
            image = cv2.imread(os.path.join(char_lib_dir, char_name), 0)
            original_size = image.shape
            image = preprocess(image)
            char = char_dic[char_id]
            self.char_lib[char_id] = {"char": char, "data": image, "size": original_size}
    
    # @analyse_it
    def match_char(self, image):
        coincidence_max, char_fit = 0, ''
        size1 = image.shape
        image = preprocess(image)
        for char_id, char_item in self.char_lib.items():
            size2 = char_item["size"]
            if 0.5 < (size1[0]/size1[1]) / (size2[0]/size2[1]) < 1.5:
                template = char_item["data"]
                image_xor = cv2.bitwise_xor(image, template)
                coincidence = image_xor[image_xor==0].shape[0] / 2500
                if coincidence_max < coincidence:
                    coincidence_max = coincidence
                    char_fit = char_item["char"]
        return char_fit, coincidence_max

    def text_word(self, cc_word):
        result_match = {}
        result_match_combine = {}
        cc_len = len(cc_word)
        cc_word.sort(key=lambda item: item[0])
        for index, cc in enumerate(cc_word):
            x1, y1, x2, y2 = cc
            char_fit, confidence = self.match_char(self.image_gray[y1:y2, x1:x2])
            result_match[tuple(cc)] = {"char": char_fit, "confidence": confidence}

            hori_range = cc[0] + 60 if index < cc_len-1 else 0
            n = 1
            index_combine = [index]
            while hori_range and cc_word[index+n][2] < hori_range:
                index_combine.append(index+n)
                n = n + 1
                if index + n >= cc_len:
                    break

            if len(index_combine) > 1:
                boxes_cc = [cc_word[index] for index in index_combine]
                cc_combine = length2coor(fuse_box(boxes_cc))
                x1, y1, x2, y2 = cc_combine
                char_fit, confidence = self.match_char(self.image_gray[y1:y2, x1:x2])
                result_match_combine[tuple(cc_combine)] = {"char": char_fit, "confidence": confidence, "cc_indexs": boxes_cc}
            # print(f'({char_fit}, {coincidence_max})', end=' ')

        inter_cc = []
        for key, result in result_match_combine.items():
            if result["confidence"] > np.mean([result_match[cc]["confidence"] for cc in result["cc_indexs"]]):
                result_match[key] = result
                for cc in result["cc_indexs"]:
                    if cc not in inter_cc:
                        inter_cc.append(cc)
            # else:
            #     cv2.imwrite(f"../else/{result['char']}_{key}.png", self.image_gray[key[1]:key[3], key[0]:key[2]])

        for cc in inter_cc:
            del(result_match[cc])

        return result_match

    # @time_it
    @analyse_it
    def text(self, options={}):
        imgInv = imu.preprocess_bw_inv(self.image_gray)
        cv2.imwrite('imgInv.jpg', imgInv)
        connected_components = get_cc('imgInv.jpg')
        os.remove('imgInv.jpg')
        connected_components = [length2coor(cc) for cc in connected_components if 5 < cc[2] < 60 and 3 < cc[3] < 60]
        # word_boxes = generate_words(connected_components, 50)
        connected_components = generate_chars(connected_components)
        word_boxes = generate_lines(connected_components, 50)
        for word_box in word_boxes:
            word_result = self.text_word(word_box)
            keys = list(word_result.keys())
            keys.sort(key=lambda item: item[0])
            for key in keys:
                result = word_result[key]
                if options.get("probability") == "true":
                    print(f'({result["char"]}, {result["confidence"]})', end=' ')
                else :
                    print(f'{result["char"]}', end=' ')
                # if result["char"]=="二":
                #     cv2.imwrite(f"二{now}.png", self.image_gray[key[1]:key[3], key[0]:key[2]])
            print()


def preprocess(image_gray, length=50):
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, (length, length), interpolation=cv2.INTER_CUBIC) #插值方式
        image_binary = imu.preprocess_bw(image_gray)
        return image_binary


if __name__ == '__main__':
    char_lib_dir = '../char_lib'
    dic_path = './word_match/char_dic_1.json'
    image = cv2.imread('../tests/page1.jpg')
    clo = CharLibOcr(char_lib_dir, dic_path, image)
    options = {}
    # options["probability"] = "true"
    clo.text(options)

