import cv2
import os

#import dis_img,use ocr to analyze word level(or other level)
# change PSM to chose other mode
import sys
sys.path.append('../packages/rectify/')
import imutils as imu
from dis_img import  ocr,RIL

def remedy(img):
        wlist = []
        dict = {}
        coor = ocr(img,RIL.WORD)
        for i in coor:
            w = i[2] - i[0]
            ax, ay, bx, by = i[0], i[1], i[2], i[3]
            wlist.append(w)

            # coordinates for each width
            dict[w] = (ax, ay, bx, by)
        nnlist = list(dict.keys())

        #sort width of these rectangles
        nlist = sorted(nnlist, reverse=True)

        # select five larege width
        fnlist = nlist[0:5]
        # print('fnlist:',fnlist)

        five_angleo = []
        angledict = {}
        for m in range(0, len(fnlist)):
            ax, ay, bx, by = dict[fnlist[m]]
            test=4
            angle = imu.get_skew_from_area(img, ax, bx, ay, by)
            #cv2.rectangle(img, (ax, ay), (bx, by), (0, 0, 255), 2)

            angleabs = abs(angle)
            angledict[angleabs] = (angle)
            five_angleo.append(angleabs)

        five_angle = sorted(five_angleo, key=lambda x: x * -1)
        # print('five_angle:',five_angle)
        if len(five_angle) < 5:
            mid_angle = angledict[five_angle[0]]
        else:
            mid_angle = angledict[five_angle[2]]
        corrected = imu.rotate_image(img, mid_angle)
        #imu.imshow_(corrected)
        return corrected

def remedy_more(imageDir, imageSaveDir):
    imagePathList = os.listdir(imageDir)
    #imagePathList=['120190703153513729.jpg']
    for i in range(0, len(imagePathList)):
        #print(imagePathList[i])
        imagePath = os.path.join(imageDir,imagePathList[i])
        img = cv2.imread(imagePath)
        corrected=remedy(img)
        cv2.imwrite(os.path.join(imageSaveDir,imagePathList[i]), corrected)



if __name__ == '__main__':
        #imageDir = 'F:/exam_dataset/waibu/waibu-liangdiangeili'
        imageDir='F:/exam/result/hough/waibu-liangdiangeili_rotate'
        imageSaveDir = 'F:/exam/result/hough_angle/ab'
        # imageSaveDir = 'F:/result/rectify_rotate'
        remedy_more(imageDir, imageSaveDir)