# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import subprocess
import math
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/ui/scv3.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

class undoCheck(object):
    def __enter__(self):
        cmds.undoInfo(openChunk=True)
    def __exit__(self, *exc):
        cmds.undoInfo(closeChunk=True)

class Window(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtUiTools.QUiLoader().load(uiFile)
        setup_ui(ui, self)
        self.connectSignal()
        self.smChk = 0
        self.prChk = 0

    def connectSignal(self):
        self.loBtn.clicked.connect(self.loadKfr)
        self.apBtn.clicked.connect(self.applyKfr)
        self.numSl.setValue(0)
        self.adSld.valueChanged.connect(self.slUpdate)
        self.movBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/mov.png")))
        self.movBtn.setIconSize(QtCore.QSize(23,23))
        self.movBtn.clicked.connect(self.playTuto)
        self.manBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/system-help.png")))
        self.manBtn.setIconSize(QtCore.QSize(23, 23))
        self.manBtn.clicked.connect(self.help)
        self.smoothBtn.clicked.connect(self.smoothCurve)
        self.smoothBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/curveEdit.png")))
        self.smoothBtn.setIconSize(QtCore.QSize(20, 20))
        self.rmPeakBtn.clicked.connect(self.removePeak)
        self.rmPeakBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/adf.png")))
        self.rmPeakBtn.setIconSize(QtCore.QSize(20, 20))
        self.rsBtn.clicked.connect(self.defK)
        self.rsBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/btis.png")))
        self.rsBtn.setIconSize(QtCore.QSize(20, 20))

    def hconv(self, text):
        return unicode(text, 'utf-8')

    def help(self):
        exp = "/usr/bin/evince"
        fileName = "/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/simplifyCurves/scvHelp.pdf"
        subprocess.Popen([exp, fileName])

    def playTuto(self):
        pdplay = "/opt/pdplayer/pdplayer64"
        fileName = "/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/simplifyCurves/scvMov.mp4"
        virEnv = os.environ.copy()
        virEnv.pop("LIBQUICKTIME_PLUGIN_DIR")
        if os.path.isfile(fileName):
            subprocess.Popen([pdplay, fileName], env=virEnv)
        else:
            QtWidgets.QMessageBox.warning(self, self.hconv("안내"), self.hconv("재생할 파일이 존재하지 않습니다."))

    def resample_keys(self, kv, thresh):
        '''
        그래프를 포인트 지점을 기준으로 쪼개는 함수.
        :param kv: 키프레임 dict() 
        :param thresh: Tolerance
        :return: thresh 값 이하로 쪼개어진 각 구간별 그래프 조각들.
        '''
        start = float(min(kv.keys()))
        end = float(max(kv.keys()))
        startv = float(kv[start])
        endv = float(kv[end])
        total_error = 0
        offender = -1
        outlier = -1
        for k, v in kv.items():
            offset = (k - start) / (end - start)
            sample = (offset * endv) + ((1 - offset) * startv)
            delta = abs(v - sample)
            total_error += delta
            if delta > outlier:
                outlier = delta
                offender = k
        if total_error < thresh or len(kv.keys()) == 2:
            return [{start: startv, end: endv}]
        else:
            rs1 = {i: j for i, j in kv.items() if i <= offender}
            rs2 = {i: j for i, j in kv.items() if i >= offender}
            return self.resample_keys(rs1, thresh) + self.resample_keys(rs2, thresh)

    def rejoin_keys(self, kvs):
        '''
        resample_keys() 함수에서 쪼갠 그래프 조각들을 다시 이어붙이는 함수
        :param kvs: 키프레임 dict()
        :return: 합쳐진 리스트
        '''
        result = {}
        for item in kvs:
            result.update(item)
        return result

    def decimate(self, keys, tolerance):
        return self.rejoin_keys(self.resample_keys(keys, tolerance))

    def loadKfr(self):
        '''
        Simplify Curve를 수행하기 위한 키 프레임 분석 및 등록.
        :return: None
        '''
        self.selAtt, self.frmList, self.sn = self.readKeyframe()
        selOb = cmds.ls(sl=True)
        self.mnK = min(cmds.keyframe(selOb, q=True, sl=True))
        self.mxK = max(cmds.keyframe(selOb, q=True, sl=True))
        if len(selOb) == 1:
            self.tgOj.setText(str(selOb[0].split(":")[1]))
        elif len(selOb) == 2:
            self.tgOj.setText(str(selOb[0].split(":")[1]) + "   " + str(selOb[1].split(":")[1]))
        else:
            self.tgOj.setText(str(selOb[0].split(":")[1]) + "   " + str(selOb[1].split(":")[1]) + "   etc")
        outF = str(int(self.mnK)) + " F ~ " + str(int(self.mxK)) + " F"
        self.frRg.setText(outF)
        self.numSl.setValue(0)
        self.numSl.setMaximum(30)
        self.adSld.setMaximum(30)

    def readKeyframe(self):
        '''
        선택한 키 프레임의 value, frame, attribute를 추출하는 함수.
        :return: selAtt(오브젝트_Attribute)
                 frmList(프레임 dict)
                 sn(키 값 dict)
        '''
        sn = dict()
        selOb = cmds.ls(sl=True)
        if cmds.keyframe(selOb, q=True, sl=True, kc=True) == 0:
            cmds.error("Select keyframes")
        else:
            temAtt = cmds.keyframe(selOb, q=True, sl=True,n=True)
            mnK = min(cmds.keyframe(selOb, q=True, sl=True))
            mxK = max(cmds.keyframe(selOb, q=True, sl=True))
            selAtt = list()
            for objAtt in temAtt:
                selAtt.append(str(objAtt))
            frmList = dict()
            for i in selAtt:
                sn[i] = dict()
                nims = cmds.listConnections(i, p=True)[0]
                obj = nims.split(".")[0]
                att = nims.split(".")[1]
                frmList[i] = cmds.keyframe(obj, at=att, t=(mnK, mxK), q=True)
                for j in range(len(frmList[i])):
                    sn[i][frmList[i][j]] = cmds.getAttr(obj + "." + att, t=frmList[i][j])
        return selAtt, frmList, sn

    def smoothCurve(self):
        '''
        그래프를 부드럽게 평균내어 주는 함수.
        현재 프레임을 K라고 할 때 K+1, K, K-1 세 프레임에서의 키 값을 평균내어
        그 평균값보다 현재 키 값이 크면 내려주고, 작으면 올려준다.
        로그 함수를 사용하여 슬라이드 입력값이 최대 99일 경우 평균값에 가깝게 이동한다.
        :return: None
        '''
        with undoCheck():
            if self.smChk == 0 and self.prChk == 0:
                selOb = cmds.ls(sl=True)
                self.mnK = min(cmds.keyframe(selOb, q=True, sl=True))
                self.mxK = max(cmds.keyframe(selOb, q=True, sl=True))
                self.selAtt, self.frmList, self.sn = self.readKeyframe()
            else:
                pass
            self.smChk = 1
            selAtt, frmList, sn = self.readKeyframe()
            for r in selAtt:
                for w in range(len(frmList[r])):
                    if w == 0 or w == (len(frmList[r]) - 1):
                        pass
                    else:
                        avr = (sn[r][frmList[r][w+1]] + sn[r][frmList[r][w]] + sn[r][frmList[r][w-1]])/3
                        adj = sn[r][frmList[r][w]] - avr
                        if adj < 0:
                            valU = sn[r][frmList[r][w]] + abs(adj)*math.log10(5)
                            cmds.setKeyframe(r, t=(frmList[r][w], frmList[r][w]), v=valU)
                        elif adj > 0:
                            valU = sn[r][frmList[r][w]] - abs(adj)*math.log10(5)
                            cmds.setKeyframe(r, t=(frmList[r][w], frmList[r][w]), v=valU)
                        else:
                            pass

    def removePeak(self):
        '''
        1프레임씩 튀는 프레임을 없애주는 함수.
        :return: None
        '''
        with undoCheck():
            sel = cmds.keyframe(q=True, n=True)
            if self.prChk == 0 and self.smChk == 0:
                selOb = cmds.ls(sl=True)
                self.mnK = min(cmds.keyframe(selOb, q=True, sl=True))
                self.mxK = max(cmds.keyframe(selOb, q=True, sl=True))
                self.selAtt, self.frmList, self.sn = self.readKeyframe()
            else:
                pass
            self.prChk = 1
            for j in sel:
                seltime = cmds.keyframe(j, q=True, tc=True, sl=True)
                for i in range(len(seltime)):
                    if seltime[i] == min(seltime) or seltime[i] == max(seltime):
                        pass
                    else:
                        adr = cmds.keyframe(j, q=True, vc=True, t=(seltime[i], seltime[i]))[0] - (cmds.keyframe(j, q=True, vc=True, t=(seltime[i+1],seltime[i+1]))[0] +cmds.keyframe(j, q=True,vc=True,t=(seltime[i-1],seltime[i-1]))[0]) / 2
                        sur = cmds.keyframe(j, q=True, vc=True, t=(seltime[i+1], seltime[i+1]))[0] - cmds.keyframe(j, q=True, vc=True, t=(seltime[i-1], seltime[i-1]))[0]
                        if abs(adr / sur) > 1:
                            getV = (cmds.keyframe(j, q=True, t=(seltime[i+1], seltime[i+1]), vc=True)[0] + cmds.keyframe(j, q=True, t=(seltime[i-1], seltime[i-1]), vc=True)[0]) / 2
                            cmds.setKeyframe(j, t=(seltime[i], seltime[i]), v=getV)
                        else:
                            pass

    def defK(self):
        '''
        self.selAtt, self.frmList, self.sn 세 변수에 저장된 값을 가지고
        키프레임이 편집되기 전의 처음 상태로 되돌리는 함수.
        :return: None 
        '''
        cmds.selectKey(cl=True)
        for i in self.selAtt:
            nims = cmds.listConnections(i, p=True)[0]
            obj = nims.split(".")[0]
            att = nims.split(".")[1]
            for j in range(len(self.frmList[i])):
                if j == 0 or j == (len(self.frmList[i]) - 1):
                    pass
                else:
                    cmds.setKeyframe(obj, at=att, t=(self.frmList[i][j], self.frmList[i][j]), v=self.sn[i][self.frmList[i][j]])
            cmds.selectKey(obj, at=att, t=(self.mnK, self.mxK), add=True)

    def slUpdate(self):
        '''
        Slide 값이 변화되면 실행되는 함수. Hold Frame의 유무에 따라 각각 다른 함수를 실행.
        :return: None
        '''
        if len(self.frRg.text()) == 0:
            return
        if self.numSl.value() == 0:
            self.defK()
            return
        holdframe = self.hdFr.text()
        if len(holdframe) != 0:
            inputHold = list()
            for j in holdframe.split(" "):
                if j.isdigit():
                    inputHold.append(int(j))
            inputHold.sort()
            del holdframe
            if min(inputHold) > self.mnK:
                if max(inputHold) < self.mxK:
                    holdframe = [int(self.mnK)] + inputHold + [int(self.mxK)]
                else:
                    cmds.warning("Invalid hold frame range.")
            else:
                cmds.warning("Invalid hold frame range.")
            spL = holdframe
        else:
            spL = None
        if spL != None:
            self.slideDrag(spL)
        else:
            self.slideDragN()

    def slideDrag(self, spL):
        '''
        Hold Frame이 있는 경우의 함수.
        :param spL: Hold frame(STR) 
        :return: None
        '''
        with undoCheck():
            self.defK()
            animCrv = cmds.keyframe(q=True, n=True)
            keyDic = dict()
            for each in animCrv:
                time = cmds.keyframe(each, q=True, tc=True, sl=True)
                value = cmds.keyframe(each, q=True, vc=True, sl=True)
                keyDic[each] = dict()
                for i in range(len(time)):
                    keyDic[each][time[i]] = value[i]
            getVlu = int(self.numSl.value())
            if getVlu < 10:
                valU = getVlu / 3.0
            elif getVlu >= 10 and getVlu < 20:
                valU = getVlu / 2.0
            elif getVlu >= 20 and getVlu < 31:
                valU = self.numSl.value()
            for ea in animCrv:
                time = cmds.keyframe(ea, q=True, tc=True, sl=True)
                keyDic[ea] = self.decimate(keyDic[ea], valU)
                cmds.cutKey(ea, t=(time[0] + 1, time[-1] - 1), clear=True)
                for i in keyDic[ea].keys():
                    cmds.setKeyframe(ea, t=(int(i), int(i)), v=keyDic[ea][i])
                for k in spL:
                    if cmds.keyframe(ea, q=True, t=(k, k)):
                        pass
                    else:
                        cmds.setKeyframe(ea, t=(int(k), int(k)), v=self.sn[ea][k])

    def slideDragN(self):
        '''
        Hold Frame이 없는 경우의 함수.
        :return: None
        '''
        with undoCheck():
            self.defK()
            animCrv = cmds.keyframe(q=True, n=True)
            keyDic = dict()
            for each in animCrv:
                time = cmds.keyframe(each, q=True, tc=True, sl=True)
                value = cmds.keyframe(each, q=True, vc=True, sl=True)
                keyDic[each] = dict()
                for i in range(len(time)):
                    keyDic[each][time[i]] = value[i]
            getVlu = int(self.numSl.value())
            if getVlu < 10:
                valU = getVlu/3.0
            elif getVlu >= 10 and getVlu < 20:
                valU = getVlu / 2.0
            elif getVlu >= 20 and getVlu <31:
                valU = self.numSl.value()
            for ea in animCrv:
                time = cmds.keyframe(ea, q=True, tc=True, sl=True)
                keyDic[ea] = self.decimate(keyDic[ea], valU)
                cmds.cutKey(ea, t=(time[0] + 1, time[-1] - 1), clear=True)
                for i in keyDic[ea].keys():
                    cmds.setKeyframe(ea, t=(int(i), int(i)), v=keyDic[ea][i])

    def applyKfr(self):
        '''
        Variables Initialize
        :return: None
        '''
        self.frRg.setText("")
        self.tgOj.setText("")
        self.hdFr.setText("")
        self.smChk = 0
        self.prChk = 0
        self.sn = dict()
        try:
            del self.selAtt
            del self.mnK
            del self.mxK
        except:
            pass
        self.numSl.setValue(0)
        cmds.selectKey(cl=True)

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