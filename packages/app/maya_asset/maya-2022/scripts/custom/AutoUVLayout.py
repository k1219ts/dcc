import pymel.core as pm
import maya.cmds as mc

import math
toolname = "NPK_UV_LAYOUT_V.1.0"

# BUILD UI
def createUI(placeholder):
    windowID = "myUVlayout"

    if mc.window(windowID, ex=True):
        mc.deleteUI(windowID, window=True)

    mc.window(windowID, title=placeholder, s=True, wh = (360,100))

    layout = mc.columnLayout("mainLayout", adj=True)

    mc.gridLayout("gridLayout1", numberOfRowsColumns = (3,3), cellWidthHeight = (120,23), parent = "mainLayout")
    mc.button(l='Get UV scale', c=lambda *args: getUVscale())
    mc.floatField('floatUVscale', value = 1, pre = 4)
    mc.button(l='Unify UV scale', c=lambda *args: unifyUVscale())
    mc.gridLayout("gridLayout2", numberOfRowsColumns = (1,1), cellWidthHeight = (360,23), parent = "mainLayout")
    mc.separator()
    mc.gridLayout("gridLayout4", numberOfRowsColumns = (1,4), cellWidthHeight = (90,23), parent = "mainLayout")
    mc.text(l='Tile Spacing')
    mc.floatField('floatTileSpacing', value = 0.025, pre = 3)
    mc.text(l='Shell Spacing')
    mc.floatField('floatShellSpacing', value = 0.025, pre = 3)
    mc.text(l='Start tile')
    mc.intField('intStartTile', value = 1)
    mc.text(l='Auto unify')
    mc.checkBox('unifyUVcheckbox', l='', value=1)
    mc.gridLayout("gridLayout7", numberOfRowsColumns = (1,1), cellWidthHeight = (360,23), parent = "mainLayout")
    mc.separator()
    mc.button(l='Layout UVs', c=lambda *args: identifyUnique())
    mc.showWindow()

def getUVscale():
    sel = mc.ls(sl=True)
    eWorldSpace = pm.PyNode(str(sel[0]) + '.e[0]').getLength(space='world')
    getEdgeUVs = mc.polyListComponentConversion(str(sel[0]) + '.e[0]', tuv=True)
    getEdgeUVsFlat = mc.ls(getEdgeUVs, fl=True)
    UVonePos = mc.polyEditUV(getEdgeUVsFlat[0], q=True, u=True)
    UVtwoPos = mc.polyEditUV(getEdgeUVsFlat[1], q=True, u=True)
    uvDistance = ((UVtwoPos[0] - UVonePos[0])**2 + (UVtwoPos[1] - UVonePos[1])**2)**0.5
    uvRatio = uvDistance/eWorldSpace / 10
    mc.floatField('floatUVscale', edit=True, value=uvRatio)

def unifyUVscale():
    sel = mc.ls(sl=True)
    for i in sel:
        unifyUV(i)

def unifyUV(objs):
    sel = [objs]
    uvRatio = mc.floatField('floatUVscale', q=True, value=True)/10
    eWorldSpace = pm.PyNode(str(sel[0]) + '.e[0]').getLength(space='world')
    getDimensions = mc.polyEvaluate(sel[0], b2=True)
    getEdgeUVs = mc.polyListComponentConversion(str(sel[0]) + '.e[0]', tuv=True)
    getEdgeUVsFlat = mc.ls(getEdgeUVs, fl=True)
    UVonePos = mc.polyEditUV(getEdgeUVsFlat[0], q=True, u=True)
    UVtwoPos = mc.polyEditUV(getEdgeUVsFlat[1], q=True, u=True)
    uvDistance = ((UVtwoPos[0] - UVonePos[0])**2 + (UVtwoPos[1] - UVonePos[1])**2)**0.5
    iUVratioWorld = uvDistance/eWorldSpace
    iUVratioWorld = uvRatio/iUVratioWorld
    iToUV = mc.polyListComponentConversion(sel[0], tuv=True)
    iToUVflat = mc.ls(iToUV, fl=True)
    qUVpivot = mc.polyEditUV(iToUVflat, q=True, u=True)
    uValSorted = sorted(qUVpivot[::2])
    vValSorted = sorted(qUVpivot[1::2])
    upivot = (uValSorted[-1:][0]+uValSorted[0])/2
    vpivot = (vValSorted[-1:][0]+vValSorted[0])/2
    mc.polyEditUV(iToUV, pu=upivot, pv=vpivot, su=iUVratioWorld, sv=iUVratioWorld)

def identifyUnique():

    sel = mc.ls(sl=True)
    allDict = {}

    for i in sel:

        faceCountPerObject = str(mc.polyEvaluate(i, f=True))
        edgeCountPerObject = str(mc.polyEvaluate(i, e=True))
        vtxCountPerObject = str(mc.polyEvaluate(i, v=True))
        allDictKey = faceCountPerObject + '_' + edgeCountPerObject + '_' + vtxCountPerObject

        if allDictKey in allDict.keys():
            allDict[allDictKey].append(i)

        if allDictKey not in allDict.keys():
            listName = [i]
            allDict[allDictKey]=listName

    layoutObjectUVs(allDict)
    return allDict


def layoutObjectUVs(selection):

    allDict = selection

    tileSpacing = mc.floatField('floatTileSpacing', q=True, v=True)
    shellSpacing = mc.floatField('floatShellSpacing', q=True, v=True)
    checkForUnification = mc.checkBox('unifyUVcheckbox', q=True, v=True)
    getStartVal = mc.intField('intStartTile', q=True, v=True)

    extraOffset = []

    for baseValue, each in enumerate(allDict.keys()):

        if checkForUnification == True:
            unifyUVscale()

        sel = allDict[each]

        selSize = len(sel)

        getDimensions = mc.polyEvaluate(sel[0], b2=True)

        dimensionsX = getDimensions[0][1] - getDimensions[0][0]
        dimensionsY = getDimensions[1][1] - getDimensions[1][0]

        uFitCal = 1 - tileSpacing*2 - dimensionsX
        maxFitU = math.trunc(uFitCal/(dimensionsX+shellSpacing)) + 1
        vFitCal = 1 - tileSpacing*2 - dimensionsY
        maxFitV = math.trunc(vFitCal/(dimensionsY+shellSpacing)) + 1

        for count in range(0, selSize, 1):

            i = sel[count]

            stackValueV = math.trunc(count/maxFitV)
            stackLoopV = count - (stackValueV*maxFitV)

            stackValueU = math.trunc(stackValueV/maxFitU)
            stackLoopU = count - stackValueU * (maxFitU * maxFitV) + 1

            uOffset = stackValueV - maxFitU * stackValueU

            iToUVflat = mc.ls(mc.polyListComponentConversion(i, tuv=True), fl=True)
            qUVpivot = mc.polyEditUV(iToUVflat, q=True, u=True)
            uValSorted = sorted(qUVpivot[::2])
            vValSorted = sorted(qUVpivot[1::2])
            upivot = (uValSorted[-1:][0]+uValSorted[0])/2
            vpivot = (vValSorted[-1:][0]+vValSorted[0])/2

            yOffset = (uOffset * (dimensionsX+shellSpacing) + stackValueU) + baseValue + sum(extraOffset) + getStartVal-1

            if yOffset  >= 10:
                yOffsetCalc = math.trunc(yOffset / 10)
                mc.polyEditUV(iToUVflat, u=-upivot+(dimensionsX/2)+tileSpacing + (uOffset * (dimensionsX+shellSpacing) + stackValueU) + baseValue + sum(extraOffset)-yOffsetCalc*10+getStartVal-1, v=-vpivot+(dimensionsY/2)+tileSpacing + stackLoopV * dimensionsY + stackLoopV * shellSpacing+yOffsetCalc)
            else:
                mc.polyEditUV(iToUVflat, u=-upivot+(dimensionsX/2)+tileSpacing + (uOffset * (dimensionsX+shellSpacing) + stackValueU) + baseValue + sum(extraOffset)+getStartVal-1, v=-vpivot+(dimensionsY/2)+tileSpacing + stackLoopV * dimensionsY + stackLoopV * shellSpacing)

            if count == selSize-1.0:
                val = selSize/(maxFitU * maxFitV)
                valCheck = selSize/(maxFitU * float(maxFitV))
                if valCheck.is_integer() == True:
                    val = val - 1
                extraOffset.append(val)