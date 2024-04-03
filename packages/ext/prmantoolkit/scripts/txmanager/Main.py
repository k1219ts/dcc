#coding:utf-8
from __future__ import print_function

#----------------------------------------------------------------------
# @ author : daeseok.chae in Dexter Studio RND
# @ date   : 2019.07.02
# @ version: 2.0 for USD Convention.
#----------------------------------------------------------------------
import sys
import os

from PySide2 import QtWidgets,QtCore,QtGui
from MainUI import Ui_Form

import txmake_proc
import tifconvert_proc

currentDir = os.path.dirname(__file__)

import platform
import glob

import ice

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Texture Management Tool")
        self.setWindowIcon(QtGui.QIcon(os.path.join(currentDir, "Resources", "icon", "appIcon.png")))

        self.ui.directoryEdit.returnPressed.connect(self.findFiles)

        self.ui.filenameList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Variant Tab
        self.ui.insertBtn.setChecked(True)

        self.ui.insertBtn.clicked.connect(self.uiUpdate)
        self.ui.changeBtn.clicked.connect(self.uiUpdate)
        self.ui.linkBtn.clicked.connect(self.uiUpdate)
        self.ui.removeBtn.clicked.connect(self.uiUpdate)
        self.uiUpdate()

        self.ui.pushButton.clicked.connect(self.executeBtnClicked)

        # Resize Tab
        self.ui.resize512Btn.clicked.connect(lambda : self.resizeBtnClicked("512"))
        self.ui.resize1KBtn.clicked.connect(lambda : self.resizeBtnClicked("1k"))
        self.ui.resize2KBtn.clicked.connect(lambda : self.resizeBtnClicked("2k"))
        self.ui.resize4KBtn.clicked.connect(lambda : self.resizeBtnClicked("4k"))
        self.ui.resize8KBtn.clicked.connect(lambda : self.resizeBtnClicked("8k"))

        self.ui.resizeProxyBtn.clicked.connect(self.proxyBtnClicked)

        # Rename Tab
        self.ui.renameExecBtn.clicked.connect(self.renameBtnClicked)

        # Metamong Tab
        self.ui.txMakeBtn.clicked.connect(self.txMakeBtnClicked)
        self.ui.texToTiffBtn.clicked.connect(self.texToTifBtnClicked)
        self.ui.texToJpgBtn.clicked.connect(self.texToJpgBtnClicked)
        self.ui.imageToJpgBtn.clicked.connect(self.imageToJpgBtnClicked)

        # Sample Tab
        self.ui.sampleExecBtn.clicked.connect(self.sampleBtnClicked)

    def srgb2lin(self, v):
        base = ice.Card(ice.constants.FLOAT, [0.0404482362771082])

        c1 = ice.Card(ice.constants.FLOAT, [12.92])
        c2 = ice.Card(ice.constants.FLOAT, [1.055])
        c3 = ice.Card(ice.constants.FLOAT, [2.4])
        c4 = ice.Card(ice.constants.FLOAT, [0.055])

        t1 = v.Add(c4).Divide(c2)
        t2 = t1.Pow(c3)

        t3 = v.Divide(c1)

        m = v.Gt(base)
        t4 = t2.Multiply(m)

        result = t4.Add(t3.Multiply(v.Le(base)))
        return result

    def lin2srgb(self, v):
        '''
        linear to srgb by image base
        :param v: pixel ice image
        :return:
        '''
        c1 = ice.Card(ice.constants.FLOAT, [1.055])
        c2 = ice.Card(ice.constants.FLOAT, [1.0 / 2.4])
        c3 = ice.Card(ice.constants.FLOAT, [12.92])

        t1 = v.Multiply(c1) # 1.055 * v
        t2 = t1.Pow(c2)     # (1.0 / 2.4) => 0.416666666667

        t3 = v.Multiply(c3)

        base = ice.Card(ice.constants.FLOAT, [0.0031308])
        m = v.Gt(base) # v > 0.0031308
        t4 = t2.Multiply(m)

        result = t4.Add(t3.Multiply(v.Le(base)))

        # if v <= 0.0031308:
        #     s = 12.92 * v
        # else:
        #     s = (pow(1.055 * v, (1.0 / 2.4))) - 0.055
        #
        # convert ice script
        # t3 = v.Multiply(12.92)
        # s = t2 - 0.055
        # base = iceColor(0.0031308)
        # m = v.Gt(base) => [0, 1, 1, 0, 0, 0, ...]
        # t2.Multiply(m)
        return result

    ######################### Rename Tab Function #########################
    def renameBtnClicked(self):
        '''
        file rename script
        :return:
        '''
        if not self.ui.filenameList.selectedItems():
            self.ErrorMsg("Select files")
            return

        if "/" in self.ui.renameOriginEdit.text() or "/" in self.ui.renameReplaceEdit.text():
            self.ErrorMsg("Can't change directory")
            return

        orgFilePathList = []
        newFilePathList = []

        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        index = 0
        for fileName in self.ui.filenameList.selectedItems():
            self.updateProgress(progDlg, index)
            index += 1
            fileName = str(fileName.text())

            originName = os.path.join(self.directoryPath, fileName)
            newName = os.path.join(self.directoryPath, fileName.replace(str(self.ui.renameOriginEdit.text()),
                                                                        str(self.ui.renameReplaceEdit.text())))

            orgFilePathList.append(originName)
            newFilePathList.append(newName)
            os.rename(originName, newName)

        progDlg.close()
        printText = "Rename Process Result\n"
        for index in range(len(newFilePathList)):
            printText += "%s\n" % (os.path.basename(newFilePathList[index]))

        self.ErrorMsg(printText, "Success")
    ######################### Rename Tab End Function #########################

    ######################### Resize Tab Function #########################
    def resizeBtnClicked(self, status):
        '''
        only resize event
        :param status:
        :return:
        '''
        resizeRule = {"proxy":512, "512":512, "1k":1024, "2k":2048, "4k":4096, "8k":8192}
        orgFilePathList = []
        newFilePathList = []

        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        index = 0
        targetDir = os.path.join(self.directoryPath, status)
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        else:
            retMsg = self.ErrorMsg("Already %s directory exists, do you want overwrite?" % status, "warning!",
                                   QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if retMsg == QtWidgets.QMessageBox.Cancel:
                return

        for fileName in self.ui.filenameList.selectedItems():
            self.updateProgress(progDlg, index)
            index += 1
            fileName = str(fileName.text())

            originFilePath = os.path.join(self.directoryPath, fileName)
            newFilepath = os.path.join(targetDir, fileName)

            if status == "proxy":
                splitExtStr = list(os.path.splitext(newFilepath))
                splitExtStr[-1] = ".jpg"
                newFilepath = "".join(splitExtStr)

            cmd = "convert {input} -resize {width} {output}".format(input=originFilePath,
                                                                    output=newFilepath,
                                                                    width=resizeRule[status])
            orgFilePathList.append(originFilePath)
            newFilePathList.append(newFilepath)
            os.system(cmd)
        progDlg.close()

        printText = "Resize %s Process Result\n" % status
        printText += "%s\n" % targetDir
        for index in range(len(newFilePathList)):
            printText += "%s\n" % (os.path.basename(newFilePathList[index]))

        self.ErrorMsg(printText, "Success")

    def proxyBtnClicked(self):
        '''
        Make Proxy
        :param status:
        :return:
        '''
        proxyResolution = 512
        newFilePathList = []

        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        index = 0
        targetDir = self.directoryPath

        # if tex convention
        targetDir = targetDir.replace("/tex/", "/proxy/")
        targetDir = targetDir.replace("/images/", "/proxy/")


        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        else:
            retMsg = self.ErrorMsg("Already proxy directory exists, do you want overwrite?", "warning!",
                                   QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if retMsg == QtWidgets.QMessageBox.Cancel:
                return

        for fileName in self.ui.filenameList.selectedItems():
            fileName = str(fileName.text())
            if not "diffC" in fileName:
                continue

            self.updateProgress(progDlg, index)
            index += 1

            # isUdim = False
            # if len(fileName.split(".")) >= 3:
            #     isUdim = True

            originFilePath = os.path.join(self.directoryPath, fileName)
            splitExtStr = list(os.path.splitext(fileName))
            splitExtStr[-1] = ".jpg"
            fileName = "".join(splitExtStr)

            newFilepath = os.path.join(targetDir, fileName)

            loadImg = ice.Load(originFilePath)
            # if isUdim:
            #     loadImg = self.srgb2lin(loadImg)

            loadImg.Save(newFilepath, ice.constants.FMT_JPEG)

            cmd = "convert {input} -resize {width} {output}".format(input=newFilepath,
                                                                    output=newFilepath,
                                                                    width=proxyResolution)

            newFilePathList.append(newFilepath)
            os.system(cmd)
        progDlg.close()

        printText = "Resize %s Process Result\n" % "proxy"
        printText += "%s\n" % targetDir
        for index in range(len(newFilePathList)):
            printText += "%s\n" % (os.path.basename(newFilePathList[index]))

        self.ErrorMsg(printText, "Success")

    ######################### Resize Tab End Function #########################

    ######################### Variant Tab Function #########################
    def findFiles(self):
        self.ui.filenameList.clear()

        # /show/wwd/asset/char/ourSoldierA/texture/pub/v01
        self.directoryPath = str(self.ui.directoryEdit.text())

        excludeList = ['Thumbs.db']

        for filename in sorted(os.listdir(self.directoryPath)):
            if filename in excludeList or filename[0] == '.' or os.path.isdir(os.path.join(self.directoryPath, filename)):
                continue
            self.ui.filenameList.addItem(filename)

    def uiUpdate(self):
        self.ui.newVariantEdit.setText("")
        self.ui.orgVariantEdit.setText("")
        self.ui.label_3.setVisible(True)
        self.ui.newVariantEdit.setVisible(True)
        self.ui.newVariantEdit.setDisabled(False)
        self.ui.orgVariantEdit.setDisabled(False)
        self.ui.newVariantEdit.setPlaceholderText("only input number [ex) 1]")
        self.ui.orgVariantEdit.setPlaceholderText("only input number [ex) 1]")

        for index in range(1, 10):
            eval("self.ui.link%s.setVisible(False)" % index)
        # eval()
        if self.ui.insertBtn.isChecked():
            self.ui.orgVariantEdit.setPlaceholderText("No input")
            self.ui.orgVariantEdit.setDisabled(True)
        elif self.ui.removeBtn.isChecked():
            self.ui.newVariantEdit.setPlaceholderText("No input")
            self.ui.newVariantEdit.setDisabled(True)
        elif self.ui.linkBtn.isChecked():
            self.ui.label_3.setVisible(False)
            self.ui.newVariantEdit.setVisible(False)
            for index in range(1, 10):
                eval("self.ui.link%s.setVisible(True)" % index)
        # elif not self.ui.linkBtn.isChecked():
        #     self.ui.newVariantEdit.setDisabled(False)

    def executeBtnClicked(self):
        if not self.ui.filenameList.selectedItems():
            self.ErrorMsg("Select files")
            return

        try:
            if not self.ui.removeBtn.isChecked() and not self.ui.linkBtn.isChecked():
                int(self.ui.newVariantEdit.text())
            if not self.ui.insertBtn.isChecked():
                int(self.ui.orgVariantEdit.text())
        except:
            self.ErrorMsg("check variant")
            return

        orgFilePathList = []
        newFilePathList = []
        if self.ui.insertBtn.isChecked():
            if not self.ui.newVariantEdit.text():
                self.ErrorMsg("check new variant")
                return

            progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
            # TODO:
            index = 0
            for item in self.ui.filenameList.selectedItems():
                self.updateProgress(progDlg, index)
                index += 1
                filepath = str(item.text())
                orgFullPath = os.path.join(self.directoryPath, filepath)

                splitFileName = orgFullPath.split('.')
                splitFileName[0] = splitFileName[0] + "_%s" % self.ui.newVariantEdit.text()
                newFullPath = ".".join(splitFileName)

                orgFilePathList.append(orgFullPath)
                newFilePathList.append(newFullPath)
                os.rename(orgFullPath, newFullPath)
            progDlg.close()
        elif self.ui.removeBtn.isChecked():
            if not self.ui.orgVariantEdit.text():
                self.ErrorMsg("check org variant")
                return

            # TODO:
            progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
            index = 0
            for item in self.ui.filenameList.selectedItems():
                self.updateProgress(progDlg, index)
                index += 1
                filepath = str(item.text())
                orgFullPath = os.path.join(self.directoryPath, filepath)

                splitFileName = orgFullPath.split('.')
                splitFileName[0] = splitFileName[0].replace("_%s" % self.ui.orgVariantEdit.text(), "")
                newFullPath = ".".join(splitFileName)

                if orgFullPath.split('.')[0] == splitFileName[0]:
                    continue

                # print "rename :", orgFullPath, newFullPath
                orgFilePathList.append(orgFullPath)
                newFilePathList.append(newFullPath)
                os.rename(orgFullPath, newFullPath)
            progDlg.close()
        elif self.ui.linkBtn.isChecked():
            if not self.ui.orgVariantEdit.text():
                self.ErrorMsg("check org variant")
                return

            checkVariant = []
            for index in range(1, 10):
                isChecked = eval("self.ui.link%d.isChecked()" % index)
                if isChecked:
                    checkVariant.append(index)

            if int(self.ui.orgVariantEdit.text()) in checkVariant:
                checkVariant.remove(int(self.ui.orgVariantEdit.text()))

            # TODO:
            progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()) * len(checkVariant))
            index = 0
            for item in self.ui.filenameList.selectedItems():
                self.updateProgress(progDlg, index)
                index += 1
                filepath = str(item.text())
                orgFullPath = os.path.join(self.directoryPath, filepath)

                for variant in checkVariant:
                    splitFileName = orgFullPath.split('.')
                    splitFileName[0] = splitFileName[0].replace("_%s" % self.ui.orgVariantEdit.text(),
                                                                "_%d" % variant)
                    newFullPath = ".".join(splitFileName)

                    if orgFullPath.split('.')[0] == splitFileName[0]:
                        continue

                    orgFilePathList.append(orgFullPath)
                    newFilePathList.append(newFullPath)

                    os.system("ln -s ./%s %s" % (os.path.basename(orgFullPath), newFullPath))
        else:
            if not self.ui.newVariantEdit.text():
                self.ErrorMsg("check new variant")
                return
            if not self.ui.orgVariantEdit.text():
                self.ErrorMsg("check org variant")
                return

            # TODO:
            progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
            index = 0
            for item in self.ui.filenameList.selectedItems():
                self.updateProgress(progDlg, index)
                index += 1
                filepath = str(item.text())
                orgFullPath = os.path.join(self.directoryPath, filepath)

                splitFileName = orgFullPath.split('.')
                splitFileName[0] = splitFileName[0].replace("_%s" % self.ui.orgVariantEdit.text(), "_%s" % self.ui.newVariantEdit.text())
                newFullPath = ".".join(splitFileName)

                if orgFullPath.split('.')[0] == splitFileName[0]:
                    continue

                orgFilePathList.append(orgFullPath)
                newFilePathList.append(newFullPath)
                if self.ui.changeBtn.isChecked():
                    os.rename(orgFullPath, newFullPath)
                elif self.ui.linkBtn.isChecked():
                    os.system("ln -s ./%s %s" % (os.path.basename(orgFullPath), newFullPath))

            progDlg.close()

        printText = "Process Result\n"
        for index in range(len(newFilePathList)):
            printText += "%s => %s\n" % (os.path.basename(orgFilePathList[index]), os.path.basename(newFilePathList[index]))

        self.ErrorMsg(printText, "Success")

        self.ui.newVariantEdit.setText("")
        self.findFiles()
    ######################### Variant Tab Function End #########################

    ######################### Metamong Tab Function Start #########################
    def txMakeBtnClicked(self):
        # @./texture/images/v001/filename.tex@
        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        filePathList = []
        for fileName in self.ui.filenameList.selectedItems():
            if ".tex" in fileName.text():
                continue
            fileName = str(fileName.text())
            filePathList.append(os.path.join(self.directoryPath, fileName))
        txMaker = txmake_proc.TxMake(filePathList)
        worker = txMaker.convert()
        isWorker = True
        value = progDlg.value()
        while (isWorker):
            isWorker = next(worker)
            value += 1
            self.updateProgress(progDlg, value)

        self.ErrorMsg("txMake Success", "Success")

    def texToTifBtnClicked(self):
        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        filePathList = []
        for fileName in self.ui.filenameList.selectedItems():
            fileName = str(fileName.text())
            filePathList.append(os.path.join(self.directoryPath, fileName))
        tifMaker = tifconvert_proc.TifMake(filePathList)
        worker = tifMaker.convert()
        isWorker = True
        value = progDlg.value()
        while (isWorker):
            isWorker = next(worker)
            value += 1
            self.updateProgress(progDlg, value)

        print("while end")

        self.ErrorMsg("tifMaker Success", "Success")

    def texToJpgBtnClicked(self):
        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        currentVersion = os.path.basename(self.directoryPath)
        textureRootDir = os.path.dirname(os.path.dirname(self.directoryPath))


        for index, fileName in enumerate(self.ui.filenameList.selectedItems()):
            fileName = str(fileName.text())
            texFileName = os.path.join(self.directoryPath, fileName)
            jpgFileName = os.path.join(textureRootDir, "images", currentVersion, fileName.replace(".tex", ".jpg"))

            loadImg = ice.Load(texFileName)
            if not os.path.exists(os.path.dirname(jpgFileName)):
                os.makedirs(os.path.dirname(jpgFileName))
            loadImg.Save(jpgFileName, ice.constants.FMT_JPEG)
            print(texFileName, jpgFileName)
            self.updateProgress(progDlg, index)

        self.ErrorMsg("jpgMaker Success", "Success")

    def imageToJpgBtnClicked(self):
        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()))
        filePathList = []
        for fileName in self.ui.filenameList.selectedItems():
            fileName = str(fileName.text())
            filePathList.append(os.path.join(self.directoryPath, fileName))

        for index, imageFile in enumerate(filePathList):
            self.updateProgress(progDlg, index)
            orgfilename = imageFile
            ext = imageFile.split('.')[-1]
            newFileName = imageFile.replace(ext, "jpg")

            loadImg = ice.Load(orgfilename)
            if not os.path.exists(os.path.dirname(newFileName)):
                os.makedirs(os.path.dirname(newFileName))
            loadImg.Save(newFileName, ice.constants.FMT_JPEG)

        self.ErrorMsg("jpgMaker Success", "Success")

    ######################### Metamong Tab Function End #########################

    ######################### Sample Tab Function Start #########################
    def sampleBtnClicked(self):
        print("sampleBtnClicked")
        channelList = []
        if self.ui.specR.isChecked():
            channelList.append("specR")
        if self.ui.specG.isChecked():
            channelList.append("specG")
        if self.ui.bump.isChecked():
            channelList.append("bump")
        if self.ui.norm.isChecked():
            channelList.append("norm")

        orgImg = ""
        if self.ui.whiteImg.isChecked():
            orgImg = "%s/Resources/icon/white.jpg" % currentDir
        elif self.ui.normImg.isChecked():
            orgImg = "%s/Resources/icon/norm.jpg" % currentDir
        elif self.ui.gray0_8.isChecked():
            orgImg = "%s/Resources/icon/gray0.8.jpg" % currentDir
        elif self.ui.gray0_5.isChecked():
            orgImg = "%s/Resources/icon/gray0.5.jpg" % currentDir
        elif self.ui.gray0_2.isChecked():
            orgImg = "%s/Resources/icon/gray0.2.jpg" % currentDir
        elif self.ui.blackImg.isChecked():
            orgImg = "%s/Resources/icon/black.jpg" % currentDir

        progDlg = self.ProgressDialog(len(self.ui.filenameList.selectedItems()) * len(channelList))
        filePathList = []
        for fileName in self.ui.filenameList.selectedItems():
            fileName = str(fileName.text())
            filePathList.append(os.path.join(self.directoryPath, fileName))

        index = 0
        for filePath in filePathList:
            # print filePath
            if "_diffC" not in filePath:
                continue

            for channel in channelList:
                newFilePath = filePath.replace("_diffC", "_%s" % channel)
                # print orgImg, "->", newFilePath
                os.system("cp -rf %s %s" % (orgImg, newFilePath))

                index += 1
                self.updateProgress(progDlg, index)

        self.ErrorMsg("jpgMaker Success", "Success")
    ######################### Sample Tab Function End #########################

    ######################### Common #########################
    def ErrorMsg(self, msgText, title = "Error", button = QtWidgets.QMessageBox.Ok):
        msgBox = QtWidgets.QMessageBox()
        msgBox.move(self.frameGeometry().center())
        msgBox.setWindowTitle(title)
        msgBox.setText(msgText)
        msgBox.setStandardButtons(button)
        return msgBox.exec_()

    def ProgressDialog(self, maximum):
        progDialog = QtWidgets.QProgressDialog()
        progDialog.setWindowTitle("Waiting...")
        progDialog.setLabelText("Progress task...")
        progDialog.setValue(0)
        progDialog.setMaximum(maximum)
        progDialog.show()
        QtWidgets.QApplication.processEvents()
        return progDialog

    def updateProgress(self, progressDlg, value):
        QtWidgets.QApplication.processEvents()
        progressDlg.setValue(value)




def main():
    app = QtWidgets.QApplication(sys.argv)
    form = MainForm()
    form.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
