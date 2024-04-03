# -*- coding: utf-8 -*-
import nuke
import random
import sys

import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def randomGray(min = .1, max = .3):
    '''
    return a random darkish gray as hex code to assign to tile and gl colours
    '''
    v = random.random() * min + (max-min)
    return int('%02x%02x%02x%02x' % (int(v*255),int(v*255),int(v*255),int(v*255)),16)

def autoBackdrop( padding=50, top = 40, fontSize = 40, text ='', widthPlus=0, nodes = None):

    '''
    auto fitting backdrop node intended for use as onCreate callback
    '''
    bd = nuke.nodes.BackdropNode()
    if not(nodes):
        nodes = nuke.selectedNodes()

    bd.setSelected( False )
    minX = [ n.xpos()  for n in nodes]
    maxX = [ n.xpos() + n.screenWidth() for n in nodes]
    minY = [ n.ypos() for n in nodes]
    maxY = [ n.ypos() + n.screenHeight()  for n in nodes]
    minX.sort(), minY.sort(), maxX.sort(), maxY.sort()

    x, y = minX[0], minY[0]
    r, t = maxX[-1], maxY[-1]
    bd.setXYpos( x-padding, y-padding-top )
    bd['label'].setValue(text)
    bd['bdwidth'].setValue( r-x + 2*padding  + widthPlus)
    bd['bdheight'].setValue( t-y +2*padding+top )
    bd['note_font_size'].setValue( fontSize )
    bd['tile_color'].setValue( randomGray() )
    return bd

def branchout():

    sn = nuke.selectedNode()
    ch = nuke.channels(sn)
    xp = sn['xpos'].value()
    yp = sn['ypos'].value() - 300

    layers = []
    valid_channels = ['red', 'green', 'blue', 'alpha', 'black', 'white']

    hslth_layers = []

    for each in ch:
        layer_name = each.split('.')[0]
        tmp = []
        for channel in ch:
            if channel.startswith(layer_name) == True:
                tmp.append(channel)
        if len(tmp) < 4:
            for i in range(4-len(tmp)):
                tmp.append(layer_name+".white")
        if tmp not in layers:
            layers.append(tmp)

    for each in layers:
        layer = each[0].split('.')[0]
        ch1 = each[0].split('.')[1]
        ch2 = each[1].split('.')[1]
        ch3 = each[2].split('.')[1]
        ch4 = each[3].split('.')[1]

        if ch1 not in valid_channels:
            ch1 = "red red"
        else:
            ch1 = ch1+" "+ch1

        if ch2 not in valid_channels:
            ch2 = "green green"
        else:
            ch2 = ch2+" "+ch2

        if ch3 not in valid_channels:
            ch3 = "blue blue"
        else:
            ch3 = ch3+" "+ch3

        if ch4 not in valid_channels:
            ch4 = "alpha alpha"
        else:
            ch4 = ch4+" "+ch4


        hslth_layers.append(layer)


    ############################################################################
    hslth_layers.sort(key=natural_keys)
    layerName = [[],[],[],
                 [],[],[],
                 [],[],[],
                 []]



    category = {'Index':layerName[0], 'fur':layerName[1], 'skin':layerName[2],
                'Combine':layerName[3], 'subsurface_':layerName[4], 'separateSpec':layerName[5],
                'Shadow':layerName[6], 'assetID':layerName[7], 'Diff':layerName[8],
                'ETC':layerName[-1]}

    categotyList = ['Mask', 'FUR', 'Skin', 'Combine', 'subsurface', 'SaperateSpec',
                    'Shadow', 'assetId', 'Diff', 'ETC']
    # reload(sys)
    #sys.setdefaultencoding('utf-8')
#    detail= {'Index_0_':u'털','Index_1_':u'몸','Index_2_':u'오른 손톱','Index_3_':u'왼 손톱',
#             'Index_4_':u'오른 발톱','Index_5_':u'왼 발톱','Index_6_':u'오른 눈','Index_7_':u'왼눈',
#             'Index_8_':u'눈 안쪽 살','Index_9_':u'치아','Index_10_':u'혀',
#             'skin_0_':u'입안','skin_1_':u'얼굴, 머리','skin_2_':u'가슴','skin_3_':u'오른손',
#             'skin_4_':u'왼손','skin_5_':u'오른발','skin_6_':u'왼발','skin_7_':u'상처',
#             'skin_8_':u'눈두덩이','skin_9_':u'손등\n발바닥흙','skin_10_':u'콧구멍',
#             'cloth_0_':u'더트부분','cloth_1_':u'로고','cloth_2_':u'메쉬','cloth_3_':u'바탕색','cloth_4_':u'무늬색',
#             'fur_0_':u'머리윗부분','fur_1_':u'양팔','fur_2_':u'등','fur_3_':u'양다리','fur_4_':u'어깨'
#             }

    createdShuffle = []
    selNode = [n.name() for n in nuke.selectedNodes()]
    for s in selNode:
      name = nuke.toNode(s).knob('file').getValue()

      for i in hslth_layers:
          if '_' in i:
              layerHead = i.split('_')[0]
          else:
              layerHead = i
          if category.get(layerHead):
              category[layerHead].append(i)
          else:
              if name.split('.')[-3].startswith('lgt'):
                if layerHead.startswith('combine'):
                    category['Combine'].append(i)
                else:
                    category['ETC'].append(i)
              else:
                  category['ETC'].append(i)

    isFirstdot = True
    for i in layerName:
        if len(i) >0:

            layerDot = nuke.nodes.Dot()
            layerDot['selected'].setValue(True)
            #if layerName.index(i) == 0:
            if isFirstdot:
                layerDot.setInput(0, sn)
                isFirstdot = False
            else:
                layerDot.setInput(0, dot2connect)
            layerDot.setYpos(int(yp) - 100)
            layerDot.setXpos(int(xp)+ 34)

            dot2connect = layerDot

            for j in i:
                shuffleDot = nuke.nodes.Dot()
                shuffleDot.setYpos(int(yp) - 100)
                shuffleDot.setXpos(int(xp) + 134)
                shuffleDot.setInput(0, dot2connect)

#                if detail.get(j):
#                    detailSticky = nuke.nodes.StickyNote()
#                    detailSticky['note_font_size'].setValue(20)
#                    detailSticky.setYpos(int(yp) - 80)
#                    detailSticky.setXpos(int(xp) + 100)
#                    detailSticky['label'].setValue(detail.get(j))

                shuffle = nuke.nodes.Shuffle()
                shuffle['in'].setValue(j)
                shuffle['red'].setValue('red')
                shuffle['green'].setValue('green')
                shuffle['blue'].setValue('blue')
                shuffle['alpha'].setValue('blue')

                shuffle.setYpos(int(yp) - 200)
                shuffle.setXpos(int(xp) + 100)
                shuffle.knob('label').setValue(j)
                shuffle['postage_stamp'].setValue(True)
                shuffle.setInput(0, shuffleDot)

                createdShuffle.append(shuffle)

                xp = xp + 100
                dot2connect = shuffleDot
                shuffle['selected'].setValue(True)
                if i.index(j) == len(i) - 1:
                    shuffleDot['note_font_size'].setValue(80)
                    shuffleDot['label'].setValue("  " + categotyList[layerName.index(i)])
                shuffleDot['selected'].setValue(True)
            xp = sn.xpos()
            yp = yp - 200
            dot2connect = layerDot
    autoBackdrop(widthPlus = 200)
    # reload(sys)
#    sys.setdefaultencoding('ascii')
