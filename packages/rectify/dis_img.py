"""
Simple script for Python3.

Use tesseract OCR to distinguish between words and pictures.

使用tesseract OCR 来区分图像和文字。

"""

# opencv
import cv2

# Batch process requires
import os

# Calculate running time
import time

# Switch locale require
import locale
from contextlib import contextmanager

from PIL import Image

# Block types
PT = ['UNKNOWN', 'PT_FLOWING_TEXT', 'PT_HEADING_TEXT', 'PT_PULLOUT_TEXT',
        'PT_EQUATION', 'PT_INLINE_EQUATION', 'PT_TABLE', 'PT_VERTICAL_TEXT',
        'PT_CAPTION_TEXT', 'PT_FLOWING_IMAGE', 'PT_HEADING_IMAGE', 'PT_PULLOUT_IMAGE',
        'PT_HORZ_LINE', 'VERT_LINE', 'NOISE', 'COUNT']


class PSM():
    OSD_ONLY = 0
    AUTO_OSD = 1
    AUTO_ONLY = 2
    AUTO = 3
    SINGLE_COLUMN = 4
    SINGLE_BLOCK_VERT_TEXT = 5
    SINGLE_BLOCK = 6
    SINGLE_LINE = 7
    SINGLE_WORD = 8
    CIRCLE_WORD = 9
    SINGLE_CHAR = 10
    SPARSE_TEXT = 11
    SPARSE_TEXT_OSD = 12
    RAW_LINE = 13
    COUNT = 14


class RIL():
    BLOCK = 0
    PARA = 1
    TEXTLINE = 2
    WORD = 3
    SYMBOL = 4


@contextmanager
def c_locale():
    """Switch locale.

    Temporarily change the operating locale to accommodate Tesseract.
    This locale is automatically finalized when used in a with-statement
    (context manager).

    """
    #print ('Enter c_locale')
    try:
        currlocale = locale.getlocale()
    except ValueError:
        #print('Manually set to en_US UTF-8')
        currlocale = ('en_US', 'UTF-8')
    #print ('Switching to C from {}'.format(currlocale))
    locale.setlocale(locale.LC_ALL, "C")
    yield
    #print ('Switching to {} from C'.format(currlocale))
    locale.setlocale(locale.LC_ALL, currlocale)



def mark_box(imagePath,coor):
    """Use a rectangular box to show what you need.

    Args:
        imagePath: File path of image.
        coor: Coordinates of box.

    Returns:
        An image object.

    """
    img = cv2.imread(imagePath)
    font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    for i in coor:
        if i[4] == 1:
            cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(0,0,255),2)
        elif i[4]==1:
            cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(0,0,255),2)
        # cv2.line(img,i[0],i[1],(255,0,0),2)
        #cv2.putText(img,str(i[0]), (j[0],j[1]), font, 2, (0,0,255), 2)
    return img


def ocr(img,level):
    """Use tesseract OCR to detection images.

    Args:
        imagePath: File path of image.
        level: Iteration level.

    Returns:
        An array with coordinate of boxes.

    """
    result = []
    with c_locale():
        from tesserocr import PyTessBaseAPI
        api = PyTessBaseAPI()
        api.SetPageSegMode(PSM.AUTO_OSD)
        # api.SetImageFile(imagePath)
        api.SetImage(Image.fromarray(img))
        blockIter = api.AnalyseLayout()
        while blockIter.Next(level):
            pt = blockIter.BlockType()
            #result.append(blockIter.Baseline(level))
            if pt in [1,6]:
              result.append(blockIter.BoundingBox(level) + (pt,))
        api.End()
    return result


def batch_process(imageDir, imageSaveDir):
    """Batch process.

    Args:
        imageDir: Source images direction.
        imageSaveDir: Destination images direction.

    Returns:
        None
    """ 
    total = 0
    imagePathList = os.listdir(imageDir)
    imagePathList.sort()
    
    startT = time.process_time()
    for i in range(0,len(imagePathList)):
        imagePath = os.path.join(imageDir, imagePathList[i])
        
        if not imagePath.endswith('.jpg'): 
            print('{} is not a jpg image, skip...'.format(imagePath))
            print ('------------------------------')
            continue

        print (imagePath)
        total += 1

        start = time.process_time()
        result = ocr(imagePath, RIL.TEXTLINE)
        end = time.process_time()

        #print ('result = %s'%(result))
        image = mark_box(imagePath, result)
        cv2.imwrite(os.path.join(imageSaveDir, imagePathList[i]), image)
        
        print ('Running time: %s Seconds'%(end - start))
        print ('------------------------------')
    endT = time.process_time()

    print('Total images: %s'%(total))
    print('Total time: %s'%(endT - startT))
    print('Average time: %s'%((endT - startT) / total))


if __name__ == '__main__':
    print ('Distinguish pictures...')

    # 单张处理

    imagePath = 'F:/exam_dataset/waibu/some/120190703153331379.jpg'
    result = ocr(imagePath,RIL.WORD)
    # print ('result = %s'%(result))
    image = mark_box(imagePath,result)
    cv2.imwrite('F:/exam//result/333.jpg', image)
    '''
    # 批量处理
    imageDir = 'F:/exam_dataset/waibu/some'
    imageSaveDir = 'F:/exam/result/try_some'
    batch_process(imageDir, imageSaveDir)
    '''
    

