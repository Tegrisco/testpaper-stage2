import cv2
import os
import sys
sys.path.append('../packages/rectify/')
import imutils as imu
from right import remedy
from rectify_hough import correct_skew2


def hough_angle(imageDir,imageSaveDir):
    imagePathList = os.listdir(imageDir)
    #imagePathList=['120190703153927345.jpg']

    for i in range(0, len(imagePathList)):
        imagePath = os.path.join(imageDir, imagePathList[i])
        print(imagePath)
        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rotated, angle = correct_skew2(img, gray)
        corrected = remedy(rotated)
        # imu.imshow_(corrected)
        cv2.imwrite(os.path.join(imageSaveDir,imagePathList[i]),corrected)


def rectify(img):
    if isinstance (img, str):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rotated, angle = correct_skew2(img, gray)
    corrected = remedy(rotated)
    return corrected

if __name__ == '__main__':
    # imageDir='F:/exam_dataset/waibu/waibu-liangdiangeili'
    imageDir = '../../math'

    # imageSaveDir = 'F:/exam/result/hough_angle/abc'
    imageSaveDir = '../../math-adjust'

    hough_angle(imageDir,imageSaveDir)