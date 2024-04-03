#coding:utf-8
#/backstage/dcc/DCC python-2 tesseract

import pytesseract
from PIL import Image

def getImageTCInfo(imgFile):
    img = Image.open(imgFile)
    # thresh = 97
    # fn = lambda x: 255 if x > thresh else 0
    # img = img.convert('L').point(fn, mode='1')
    crop = img.crop((0, 0, img.width, img.height))

    # whiteList = '0123456789_:/ABCDEFGHIJKLMNOPQRSTUVWXYZ \[\].abcdefghijklmnopqrtuvwxyz'
    text = pytesseract.image_to_string(crop, lang='eng+hel', config='--oem 1 --psm 3')

    img.close()
    return text

    # clipList = []
    # for sepText in text.split('\n'):
    #     if sepText:
    #         print sepText
    #         clip, tc = sepText.split(' ')
    #         clipList.append((clip, tc.split('F:')[-1]))
    #
    # return clipList