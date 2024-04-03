# -*- coding: utf-8 -*-

from PySide2 import QtCore, QtGui, QtWidgets
import sys
from PySide2.QtWidgets import *
import os
dir_path =os.environ.get('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS')
setFile = ''
if dir_path :
    setFile = dir_path.split('\n')[0]

def groomVersion() :
    res = os.listdir(setFile)
    for file in res:
        under = setFile + '/' + file
        if 'groom' in under:
            grmver = os.listdir(under)
            for gver in grmver:
                ttt = (under + '/' + gver)


                if 'scenes' in ttt :
                    grmscenes = ttt

                    return grmscenes

def preview():
    res = os.listdir(setFile)
    for file in res :
        under = setFile + '/' + file
        if 'preview' in under :
            previewFold = under

            return previewFold

def modelVersion() :
    res = os.listdir(setFile)
    for file in res:
        under = setFile + '/' + file
        if 'model' in under:
            modelver = os.listdir(under)
            for mver in modelver:
                ttt = (under + '/' + mver)

                if 'v0' in ttt :
                    lastfold = []
                    gentimemodel = os.path.getctime(ttt)
                    lastfold.append((ttt, gentimemodel))
            lastVerMo = max(lastfold, key=lambda  x: x[1])[0]

            return lastVerMo

def texVersion():
    # res = os.listdir(setFile)

    # if 'texture' not in res :
    #     os.system("mkdir {0}/texture".format(setFile))

    res = os.listdir(setFile)

    for file in res:
        under = setFile + '/' + file

        if 'texture' in under:
            # modelver = os.listdir(under)

            # if 'images' not in modelver:
            #     os.system("mkdir {0}/images".format(under))
            #     os.system("mkdir {0}/images/v001".format(under))
            modelver = os.listdir(under)
            for tver in modelver:
                image = under +'/'+ tver+'/'
                if 'images' in image :
                    endgenTime = []
                    for time in os.listdir(image):              # lastTime folder
                        dirPath = image + time
                        gentime = os.path.getctime(dirPath)
                        endgenTime.append((dirPath, gentime))
                    lastVer = max(endgenTime, key=lambda x: x[1])[0]


                    return lastVer

mdir = modelVersion()
asName = filter(None, mdir.split('/'))

gdir = groomVersion()
if gdir :
    grName = filter(None, gdir.split('/'))

tdir = texVersion()
if tdir :
    txName = filter(None, tdir.split('/'))

pdir = preview()
if pdir :
    prName = filter(None, pdir.split('/'))

outdir = "/knot/show/asset/{0}/{1}/model/".format(asName[4],asName[1])

asset = ('/knot/show/asset/{}'.format(asName[4]))
showName = ('{0}/{1}'.format(asset, asName[1]))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("AssetPublish")
        MainWindow.resize(500, 560)
        MainWindow.setStyleSheet("background-color: rgb(80,80,80);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setStyleSheet("color : white; font : bold 14px;")
        self.lineEdit.setText(outdir)
        self.gridLayout.addWidget(self.lineEdit, 1, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        self.gridLayout.addWidget(self.pushButton, 3, 0, 1, 1)
        self.treeWidget = QtWidgets.QTreeWidget(self.centralwidget)
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.headerItem().setText(0, "1")
        self.treeWidget.setStyleSheet("""
                                    QTreeWidget { color: white; font: bold 16px;}
                                    QHeaderView { color: white; font: bold 16px;}
                                    """)
        self.treeWidget.setColumnCount(2)
        self.treeWidget.setHeaderLabels(["asset", "version"])
        self.treeWidget.header().setSectionResizeMode(QHeaderView.Stretch)
        self.TreeWidgetItem()

        self.treeWidget.expandAll()
        self.gridLayout.addWidget(self.treeWidget, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setStyleSheet("color: white;")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def TreeWidgetItem(self):
        if gdir :
            self.item = QTreeWidgetItem(self.treeWidget)
            self.item.setText(0, 'groom')
            self.item.setCheckState(0, QtCore.Qt.Checked)

            self.childitem = QTreeWidgetItem(self.item)
            self.childitem.setText(0, "{0}".format(grName[4]))
            self.childitem.setText(1, "{0}".format(grName[-1]))
            self.childitem.setTextColor(1, 'blue')

        self.item2 = QTreeWidgetItem(self.treeWidget)
        self.item2.setText(0, 'model')
        self.item2.setCheckState(0, QtCore.Qt.Checked)

        self.childitem2 = QTreeWidgetItem(self.item2)
        self.childitem2.setText(0, "{0}".format(asName[4]))
        self.childitem2.setText(1, "{0}".format(asName[-1]))
        self.childitem2.setTextColor(1,'blue')

        if pdir :
            self.item3 = QTreeWidgetItem(self.treeWidget)
            self.item3.setText(0, 'preview')
            self.item3.setCheckState(0, QtCore.Qt.Checked)
            self.childitem3 = QTreeWidgetItem(self.item3)
            self.childitem3.setText(0, "{0}".format(asName[4]))

            self.childitem3.setText(1, "{0}".format(prName[-1]))
            self.childitem3.setTextColor(1, 'blue')


        if tdir :
            self.item4 = QTreeWidgetItem(self.treeWidget)
            self.item4.setText(0, 'texture')
            self.item4.setCheckState(0, QtCore.Qt.Checked)
            self.childitem4 = QTreeWidgetItem(self.item4)
            self.childitem4.setText(0, "{0}".format(txName[-2]))

            self.childitem4.setText(1, "{0}".format(txName[-1]))
            self.childitem4.setTextColor(1, 'blue')


    def pushButtonClicked(self):

        if self.item2.checkState(0) == QtCore.Qt.Checked :
            self.makeDirmodel()
            os.system("cp -r " + mdir + "/* /knot/show/asset/{0}/{1}/model/".format(asName[4], asName[1]))
        else :
            pass
        if gdir :
            if self.item.checkState(0) == QtCore.Qt.Checked :
                self.makeDirgroom()
                os.system("cp -r " + gdir + "/* /knot/show/asset/{0}/{1}/groom/".format(asName[4], asName[1]))
            else :
                pass

        if pdir :
            if self.item3.checkState(0) == QtCore.Qt.Checked :
                self.makeDirtexture()
                os.system("cp -r " +tdir+ "/* /knot/show/asset/{0}/{1}/texture/".format(asName[4], asName[1]))
            else :
                pass

        if tdir :
            if self.item4.checkState(0) == QtCore.Qt.Checked :
                self.makeDirtexture()
                os.system("cp -r " +tdir+ "/* /knot/show/asset/{0}/{1}/texture/".format(asName[4], asName[1]))
            else :
                pass

        if self.item.checkState(0) == QtCore.Qt.Unchecked and self.item2.checkState(0) == QtCore.Qt.Unchecked and self.item3.checkState(0) == QtCore.Qt.Unchecked and self.item4.checkState(0) == QtCore.Qt.Unchecked :
            self.msg = QMessageBox()
            self.msg.setWindowTitle("publish")
            self.msg.setText('Nothing !')

            self.msg.exec_()
        else :
            self.msg = QMessageBox()
            self.msg.setWindowTitle("publish")
            self.msg.setText('Publish Complate ! ')

            self.msg.exec_()

        pass



    def makeDirmodel(self):

        os.system("mkdir {0}".format(asset))
        os.system("mkdir {}".format(showName))
        os.system("mkdir {}/model".format(showName))

    def makeDirgroom(self):

        os.system("mkdir {}/groom".format(showName))

    def makeDirtexture(self):

        os.system("mkdir {}/texture".format(showName))

    def makePreview(self):
        os.system(("mkdir {}/preview".format(showName)))

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "AssetPublish", None, -1))
        self.pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "Publish", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("MainWindow", "outDir", None, -1))



