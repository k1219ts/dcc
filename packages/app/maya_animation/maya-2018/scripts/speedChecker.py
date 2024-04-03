# -*- coding: utf-8 -*-

import maya.cmds as cmds
import math
from functools import partial

def speedChecker(*args):
    if cmds.window('speedWin', q=1, ex=1):
        cmds.deleteUI('speedWin')
    cmds.window('speedWin', title='Speed')
    cmds.columnLayout(w=160, h=100, rowSpacing=5)
    cmds.rowColumnLayout(numberOfColumns=2)
    cmds.text('Scene Scale    1 : ')
    cmds.floatField('fieldA', w=50, h=20, value=1)
    cmds.columnLayout(rowSpacing=2, adj=1)
    cmds.button('showSpeed', h=50, label='Show Speed', c=partial(speedHud))
    cmds.button('hideSpeed', label='Hide Speed', c=partial(delSpeedHud))
    cmds.showWindow()

def getSpeed(target, *args):
    try:
        sceneScale = cmds.floatField('fieldA', q=1, v=1)
        currentFrame = cmds.currentTime(q=1)
        beforePos = cmds.getAttr(target[0]+'.worldMatrix', t=currentFrame-1)
        currentPos = cmds.getAttr(target[0]+'.worldMatrix', t=currentFrame)
        distance = math.sqrt(math.pow(beforePos[12]-currentPos[12], 2) + math.pow(beforePos[13]-currentPos[13], 2) + math.pow(beforePos[14]-currentPos[14], 2))
        speed = (distance/1.157)/sceneScale
        showSpeed = '%d km/h'%speed
        return showSpeed
    except RuntimeError:
        cmds.confirmDialog(title='확인', b='확인', message='HUD를 hide 해 주세요.')
def speedHud(*args):
    global speedTarget

    listHead = cmds.headsUpDisplay( lh = 1, q = 1 )
    if listHead:
        for hList in listHead:
            cmds.headsUpDisplay(hList, rem = 1)

    speedTarget = cmds.ls(sl=1)
    if cmds.headsUpDisplay('speedHud', q=1, ex=1):
        cmds.headsUpDisplay('speedHud', rem=1)
    if speedTarget:
        cmds.headsUpDisplay('speedHud', section=7, block=3, label='Speed', command=partial(getSpeed,speedTarget), atr=1)
    else:
        cmds.confirmDialog(title='확인', b='확인',
                                message='하나의 객체를 선택해 주세요.')

def delSpeedHud(*args):
    if cmds.headsUpDisplay('speedHud', q=1, ex=1):
        cmds.headsUpDisplay('speedHud', rem=1)
