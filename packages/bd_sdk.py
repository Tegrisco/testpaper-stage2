
'''
Script for Python3.

Bai Du SDK demo.

百度sdk识别工具包.

'''

# Bai Du SDK
from aip import AipOcr

# opencv
import cv2

# numpy
import numpy as np


def bd_access():
    APP_ID = '15722700'
    API_KEY = 'Y88PlnEhTgdhPy2GZwqHpRKL'
    SECRET_KEY = 'WLNNoQSebTr9TQBXxMerG4MVVustMGLf'

    # APP_ID = '17110215'
    # API_KEY = 'V4gAVqk2b4XLnbTDEBPSiGRV'
    # SECRET_KEY = 'F2IKGs3GPDpHzwlgxob3gvVhwHQdQteT'

    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    return client


def bd_rec(imgSrc, client, imgForm='.jpg', api='accurate'):
    '''可选参数
    options = {}
    options["recognize_granularity"] = "big"
    options["language_type"] = "CHN_ENG"
    options["detect_direction"] = "true"
    options["detect_language"] = "true"
    options["vertexes_location"] = "true"
    options["probability"] = "true"
    '''
    options = {}
    options["recognize_granularity"] = "small"
    imgEncode = cv2.imencode(imgForm, imgSrc)[1]
    dataEncode = np.array(imgEncode)
    strEncode = dataEncode.tostring()
    if api == 'accurate':
        result = client.accurate(strEncode, options)
    else:
        result = client.general(strEncode, options)
    return result

if __name__ == '__main__':
    img = cv2.imread('./img/3.jpg')
    client = bd_access()
    print(bd_rec(img, client, api='accurate'))




