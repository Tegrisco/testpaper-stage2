#!/usr/bin/python3
# coding = UTF-8

"""
Simple script for Python3.

Use open source Tesseract OCR API.

Get text from an image.

对图片进行文字识别。


"""

# Test running time
import time

# Switch locale require
import locale
from contextlib import contextmanager


@contextmanager
def c_locale():
    """Switch locale.

    Temporarily change the operating locale to accommodate Tesseract.
    This locale is automatically finalized when used in a with-statement
    (context manager).

    Args: None
    Returns: None
    Raises: None

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



def tess_ocr(img):
    """Get text from an image.

    Args:
        img: The file path of image.

    Returns:
        A string.
    Raises:
        IOError: An error occurred accessing the img object.

    """
    with c_locale():
        from tesserocr import PyTessBaseAPI, PSM
        api = PyTessBaseAPI(lang='chi_sim', psm=PSM.AUTO_OSD)
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        api.End()
    return text


if __name__ == '__main__':
    img = '/Users/zhengzhihuang/projects/testpaper/img/questions2/all/1-1_5.png'
    start = time.process_time()
    text = tess_ocr(img).replace(' ', '')
    text_list = text.split('\n')
    end = time.process_time()
    # print('Text:\n%s'%(text))
    print(text_list)
    print('Running time: %s'%(end-start))




