# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import maya.mel as mel
from PySide2 import QtWidgets, QtCore, QtGui, QtCompat
axisList = ["+z", "-z", "+x", "-x"]

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/attachCurve.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.axisCB.addItems(axisList)
        self.connectSignal()

    def connectSignal(self):
        self.goBtn.clicked.connect(self.doIt)
        self.getSpanBtn.clicked.connect(self.getSpans)
        self.crvBtn.clicked.connect(self.atCurve)

    def doIt(self):
        selectItems = cmds.ls(sl=True)
        nsChar = selectItems[0].split(":")[0]
        rt = str(cmds.listRelatives(selectItems[0], p=True, f=True)[0].split("|")[1])
        minBound = cmds.getAttr(rt + ".boundingBoxMin")[0]
        maxBound = cmds.getAttr(rt + ".boundingBoxMax")[0]
        curveZeroPosition = cmds.pointPosition(selectItems[-1] + ".cv[0]", w=True)
        axis = str(self.axisCB.currentText())
        if axis == "+z":
            addofs = abs((maxBound[2] - minBound[2]) / 12)
            ofs = abs(cmds.xform(nsChar + ":place_CON", q=True, wd=True, t=True)[2] - minBound[2])
            atp = [curveZeroPosition[0], curveZeroPosition[1], curveZeroPosition[2] + ofs + addofs]
        elif axis == "-z":
            addofs = abs((maxBound[2] - minBound[2]) / 12)
            ofs = abs(maxBound[2] - cmds.xform(nsChar + ":place_CON", q=True, wd=True, t=True)[2])
            atp = [curveZeroPosition[0], curveZeroPosition[1], curveZeroPosition[2] - ofs - addofs]
        elif axis == "+x":
            addofs = abs((maxBound[0] - minBound[0]) / 12)
            ofs = abs(cmds.xform(nsChar + ":place_CON", q=True, wd=True, t=True)[0] - minBound[0])
            atp = [curveZeroPosition[0] + ofs + addofs, curveZeroPosition[1], curveZeroPosition[2]]
        elif axis == "-x":
            addofs = abs((maxBound[0] - minBound[0]) / 12)
            ofs = abs(maxBound[0] - cmds.xform(nsChar + ":place_CON", q=True, wd=True, t=True)[0])
            atp = [curveZeroPosition[0] - ofs - addofs, curveZeroPosition[1], curveZeroPosition[2]]
        cmds.xform(nsChar + ":place_CON", wd=True, t=atp)
        gndY = cmds.xform(nsChar + ":place_CON", q=True, wd=True, t=True)[1]
        csValue = int(self.csLE.text())
        crsValue = int(self.crsLE.text())
        cvOrig = self.curveAttach(selectItems[0], gndY, csValue, crsValue)  #
        self.curveSet(selectItems[-1], axis)    #
        cmds.wire(cvOrig, gw=False, en=1, dds=(0, 10), ce=0, li=0, w=str(selectItems[1]))
        orgMinT = cmds.playbackOptions(q=True, min=True) + 1
        cmds.currentTime(orgMinT)
        self.mkCtrl(selectItems[-1])
        cmds.playbackOptions(ast=orgMinT, min=orgMinT)

    def getSpans(self):
        sel = str(cmds.ls(sl=True)[0])
        spNum = cmds.getAttr(sel + ".spans")
        self.crsLE.setText(str(spNum))

    def atCurve(self):
        axis = str(self.axisCB.currentText())
        sel = cmds.ls(sl=True)
        zeroPosition = cmds.pointPosition(str(sel[-1]) + ".cv[0]")
        nsChar = sel[0].split(":")[0]
        selRoot = str(cmds.listRelatives(nsChar + ":place_CON", p=True, f=True)[0].split("|")[1])
        if axis == "+z":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", 0)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0], zeroPosition[1], zeroPosition[2] - bs[2] - bs[2] / 10]
        elif axis == "-z":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", 180)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0], zeroPosition[1], zeroPosition[2] + bs[2] + bs[2] / 10]
        elif axis == "+x":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", 90)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0] - bs[0] - bs[0] / 10, zeroPosition[1], zeroPosition[2]]
        elif axis == "-x":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", -90)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0] + bs[0] + bs[0] / 10, zeroPosition[1], zeroPosition[2]]
        newCurve = cmds.curve(p=[newCurveZeroP, zeroPosition], d=1, ws=True)
        cmds.rebuildCurve(newCurve, rt=0, end=1, d=3, kr=1, s=16, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)
        cmds.attachCurve(newCurve, sel[-1], ch=0, bb=0.5, kmk=1, m=0, bki=0, p=0.1, rpo=0)
        cmds.setAttr(sel[-1] + ".visibility", 0)
        cmds.setAttr(newCurve + ".visibility", 0)
        mel.eval("DeleteHistory;")
        cmds.select(sel)

    def trChange(self, trList, val):  # trList : 각 컨트롤러 위치값 리스트, val : 뼈의 분할 수
        '''
        :param trList: Translate data list of each controllers 
        :param val: Joint numbers to divide
        :return newTrList : Translate data list of divided joints
        '''
        newTrList = list()
        for i in range(len(trList)):
            if i == 0:
                pass
            else:
                for k in range(val - 1):
                    sub = [((trList[i][0] - trList[i - 1][0]) / val) * (k + 1),
                           ((trList[i][1] - trList[i - 1][1]) / val) * (k + 1),
                           ((trList[i][2] - trList[i - 1][2]) / val) * (k + 1)]
                    pos = [trList[i - 1][0] + sub[0], trList[i - 1][1] + sub[1], trList[i - 1][2] + sub[2]]
                    newTrList.append(pos)
            newTrList.append(trList[i])
        return newTrList

    def checkCtrl(self, sel):
        '''
        :param sel: Controller name what have iterable number 
        :return conList: Controller name list
        '''
        if sel.split("_")[-2][-2].isdigit():
            conList = cmds.ls("_".join(sel.split("_")[:-2]) + "_" + sel.split("_")[-2][:-2] + "*_" + sel.split("_")[-1])
        elif sel.split("_")[-2][-1].isdigit():
            conList = cmds.ls("_".join(sel.split("_")[:-2]) + "_" + sel.split("_")[-2][:-1] + "*_" + sel.split("_")[-1])
        else:
            conList = [sel]
        cmds.select(conList)
        result = cmds.confirmDialog(t="Confirm", message="Check Controllers",
                                    button=["OK", "Cancel"], defaultButton="OK", cancelButton="Cancel")
        if result == "OK":
            return conList
        else:
            return 0

    def curveAttach(self, sel, gndY, val, rbp):
        '''
        :param sel: A body controller name what have iterable number
        :param gndY: Ground translateY value 
        :param val: Joint numbers to divide
        :param rbp: IK-Handle curve spans value
        :return: ikh[-1] : IK-Handle curve name 
        '''
        conList = self.checkCtrl(sel)
        jntList = list()
        trList = list()
        for i in range(len(conList)):
            tr = cmds.xform(conList[i], q=True, ws=True, t=True)
            trList.append(tr)
        # input Joints Number
        if val == 1:
            pass
        else:
            trList = self.trChange(trList, val)
        cmds.select(cl=True)
        # Make Joints
        for j in range(len(trList)):
            jnt = str((cmds.joint(p=(trList[j][0], gndY, trList[j][2]))))
            jntList.append(jnt)
            if j != 0:
                cmds.joint(jntList[j - 1], e=True, zso=True, oj='xyz', sao="yup")
        # IK-Handle connect & constraint
        ikh = cmds.ikHandle(sj=jntList[0], ee=jntList[-1], sol="ikSplineSolver", scv=False, roc=True, ccv=True, pcv=True)
        spNum = cmds.getAttr(ikh[2] + ".spans") + 3
        ds = abs(cmds.pointPosition(ikh[2] + ".cv[" + str(spNum - 1) + "]", w=True)[2] - cmds.pointPosition(ikh[2] + ".cv[0]", w=True)[2])
        cmds.move(0, 0, -ds / 10, ikh[2] + ".cv[" + str(spNum - 1) + "]", r=True, os=True, wd=True)
        cmds.rebuildCurve(ikh[2], rt=0, end=1, d=3, kr=1, s=spNum - 3 + rbp, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)
        for e in range(len(conList)):
            nul = cmds.listRelatives(conList[e], p=True, c=False)[0]
            cmds.pointConstraint(jntList[e * val], nul, w=1, mo=False)
            cmds.orientConstraint(jntList[e * val], nul, w=1, mo=True)
        return ikh[-1]

    def curveSet(self, sel, axis):
        '''
        :param sel: Path Curve name
        :param axis:  Axis direction
        :return: None
        '''
        selshp = str(cmds.listRelatives(sel, p=False, c=True)[0])
        sp = cmds.getAttr(sel + ".spans") + 3
        minT = cmds.playbackOptions(q=True, min=True)
        cmds.setKeyframe(sel + ".cv[0:%s]" % str(sp - 1), t=(minT, minT))
        cmds.currentTime(minT - 1)
        ap = cmds.pointPosition(sel + ".cv[2]", w=True)
        bp = cmds.pointPosition(sel + ".cv[0]", w=True)
        dis = ((bp[0] - ap[0]) ** 2 + (bp[1] - ap[1]) ** 2 + (bp[2] - ap[2]) ** 2) ** 0.5
        for i in range(sp):
            if i == 0:
                pass
            else:
                dsv = i * dis
                zeroPntX = cmds.getAttr(selshp + ".controlPoints[0].xValue")
                zeroPntY = cmds.getAttr(selshp + ".controlPoints[0].yValue")
                zeroPntZ = cmds.getAttr(selshp + ".controlPoints[0].zValue")
                if axis == "+x":
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].xValue", zeroPntX + dsv)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].yValue", zeroPntY)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].zValue", zeroPntZ)
                elif axis == "+z":
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].xValue", zeroPntX)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].yValue", zeroPntY)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].zValue", zeroPntZ + dsv)
                elif axis == "-x":
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].xValue", zeroPntX - dsv)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].yValue", zeroPntY)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].zValue", zeroPntZ)
                elif axis == "-z":
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].xValue", zeroPntX)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].yValue", zeroPntY)
                    cmds.setAttr(selshp + ".controlPoints[" + str(i) + "].zValue", zeroPntZ - dsv)
        cmds.setKeyframe(sel + ".cv[0:%s]" % str(sp - 1), t=(minT - 1, minT - 1))
        cmds.playbackOptions(min=minT - 1)

    def mkCtrl(self, sel):
        '''
        :param sel: Curve name 
        :return: None
        '''
        a = cmds.getAttr(sel + '.spans') + 3
        shp = str(cmds.listRelatives(sel, c=True, type="shape")[0])
        g1 = cmds.group(n='curveF_attach_GRP', em=True)
        g2 = cmds.group(n='curveF_crtl_GRP', em=True)
        ctrList = list()
        for i in range(a):
            b = cmds.xform(sel + '.cv[' + str(i) + ']', q=True, t=True, ws=True)
            sph_ = cmds.curve(p=[(0, 0, 0.5), (0.353554, 0, 0.353554), (0.5, 0, 0), (0.353554, 0, -0.353554), (0, 0, -0.5),(-0.353554, 0, -0.353554), (-0.5, 0, 0), (-0.353554, 0, 0.353554), (0, 0, 0.5), (0, 0.25, 0.433013),(0, 0.433013, 0.25), (0, 0.5, 0), (0, 0.433013, -0.25), (0, 0.25, -0.433013), (0, 0, -0.5),(0, -0.25, -0.433013), (0, -0.433013, -0.25), (0, -0.5, 0), (0, -0.433013, 0.25),(0, -0.25, 0.433013), (0, 0, 0.5), (0.353554, 0, 0.353554), (0.5, 0, 0), (0.433013, 0.25, 0),(0.25, 0.433013, 0), (0, 0.5, 0), (-0.25, 0.433013, 0), (-0.433013, 0.25, 0), (-0.5, 0, 0),(-0.433013, -0.25, 0), (-0.25, -0.433013, 0), (0, -0.5, 0), (0.25, -0.433013, 0),(0.433013, -0.25, 0), (0.5, 0, 0)], d=1)
            sph_nul = cmds.group(sph_)
            loc_ = str(cmds.spaceLocator()[0])
            new_loc = cmds.rename(loc_, 'pathAttach' + str(i) + '_LOC')
            locShape_ = str(cmds.listRelatives(new_loc, type="shape")[0])
            new_sph = cmds.rename(sph_, 'pathCurve' + str(i) + '_CON')
            ctrList.append(new_sph)
            new_sphNul = cmds.rename(sph_nul, 'pathCurve' + str(i) + '_NUL')
            cmds.move(b[0], b[1], b[2], new_loc)
            mtm = str(cmds.shadingNode('multMatrix', asUtility=1))
            cmds.connectAttr(locShape_ + '.worldMatrix[0]', mtm + '.matrixIn[0]', force=1)
            cmds.connectAttr(shp + '.parentInverseMatrix[0]', mtm + '.matrixIn[1]', f=1)
            dcm = str(cmds.shadingNode('decomposeMatrix', asUtility=1))
            cmds.connectAttr(mtm + '.matrixSum', dcm + '.inputMatrix', force=1)
            cmds.connectAttr(dcm + '.outputTranslate', shp + '.controlPoints[' + str(i) + ']', f=1)
            cmds.move(b[0], b[1], b[2], new_sphNul)
            cmds.parentConstraint('pathCurve' + str(i) + '_CON', new_loc)
            cmds.parent(new_loc, g1)
            cmds.parent(new_sphNul, g2)
            cmds.hide(new_loc)
            cmds.setAttr(new_sph + '.overrideEnabled', 1)
            cmds.setAttr(new_sph + '.overrideColor', 22)
            cmds.cluster(new_sph)
            cmds.delete(new_sph, ch=True)
        cmds.addAttr(g2, ln="Controller_Scale", at="short")
        for j in ctrList:
            cmds.connectAttr(g2 + ".Controller_Scale", j + ".scaleX")
            cmds.connectAttr(g2 + ".Controller_Scale", j + ".scaleY")
            cmds.connectAttr(g2 + ".Controller_Scale", j + ".scaleZ")
        cmds.setAttr(g2 + ".Controller_Scale", 7)
        attLst = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ", "visibility"]
        for k in attLst:
            cmds.setAttr(g2 + '.' + k, k=False)
        cmds.setAttr(g2 + '.Controller_Scale', k=True)

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()