# -*- coding:utf-8 -*-
__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import maya.OpenMayaUI as mui
import os
import json

from PySide import QtCore, QtGui
import pysideuic
import xml.etree.ElementTree as xml
from cStringIO import StringIO

import BakeSequencer.BakeSeqModule_A as BSM
reload(BSM)
import customWrapinstance

# ======================================================================================================================= #

currentpath = os.path.abspath( __file__ )
UIROOT = os.path.dirname(currentpath)
UIFILE = os.path.join(UIROOT, "BakeSeq.ui")
css = open( os.path.join(UIROOT, 'darkorange.css'), 'r' ).read()
rcss = css.replace("RELATIVE_PATH", UIROOT)
windowObject = "Bake Sequencer v1.0"
dockMode = False

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return customWrapinstance.wrapinstance(long(ptr), QtGui.QWidget)
    #return shiboken.wrapInstance(long(ptr), QtGui.QWidget)


def loadUiType(uiFile):
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec pyc in frame

        form_class = frame['Ui_%s' % form_class]
        base_class = eval('QtGui.%s' % widget_class)

    return form_class, base_class


formclass, baseclass = loadUiType(UIFILE)


class BakeSequencer(formclass, baseclass):
    def __init__(self, parent=getMayaWindow()):
        super(BakeSequencer, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(windowObject)
        self.setStyleSheet(rcss)
        self.connectSlot()
        self.updateData()

    def connectSlot(self):
        self.DoIt_pushButton.clicked.connect(self.DoIt)
        self.All_checkBox.stateChanged.connect(self.updateData)

    def getSeqList(self):

        if self.All_checkBox.isChecked():
            self.SeqList = AllSeqList
        else:
            self.SeqList = self.seqList_comboBox.currentText()

    def updateData(self):
        self.AllSeqList = cmds.ls(type="sequencer")
        self.seqList_comboBox.clear()

        if self.All_checkBox.isChecked():
            self.seqList_comboBox.setEnabled(False)
        else:
            self.seqList_comboBox.setEnabled(True)
            self.seqList_comboBox.addItems(self.AllSeqList)


    def DoIt(self):
        if self.All_checkBox.isChecked():
            self.SeqList = self.AllSeqList
        else:
            self.SeqList = list( str(self.seqList_comboBox.currentText()) )

        ALLOBJ = BSM.bakeAllKeys()
        SEQ_DATA = BSM.writeSeqData(self.SeqList)
        SDFI = BSM.writeseqDataFrameIndexed(SEQ_DATA)
        BSM.applyTimewarp(SEQ_DATA, SDFI, ALLOBJ)

def runUI():
    global BakeSeq
    try:
        BakeSeq.close()
    except:
        pass

    BakeSeq = BakeSequencer()

    if dockMode:
        cmds.dockControl(label=windowObject, area="right", content=BakeSeq, allowedArea=["left", "right"])
    else:
        BakeSeq.show()
        BakeSeq.resize(400, 100)
