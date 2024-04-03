import sys, os
import nuke, nukescripts

from PySide2 import QtWidgets, QtCore, QtGui

from ui_allPlatesImport import Ui_Form
import random


class MainWidget_v02(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent, QtCore.Qt.WindowStaysOnTopHint)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #self.listWidget = DirectoryView(self)
        self.listWidget = QtWidgets.QListWidget(self)
        self.listWidget.setGeometry(QtCore.QRect(10, 30, 631, 241))
        self.listWidget.setObjectName("listWidget")

        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget.setViewMode(QtWidgets.QListView.IconMode)
        self.listWidget.folderIcon = QtWidgets.QFileIconProvider().icon(QtWidgets.QFileIconProvider.Folder)
        self.listWidget.folderIcon = QtGui.QIcon(self.listWidget.folderIcon.pixmap(100, 100))


        self.ui.searchButton.clicked.connect(self.listUp)


        self.ui.pushButton.clicked.connect(self.openSequence)
        self.ui.pushButton_2.clicked.connect(self.close)

    def listUp(self):

        self.listWidget.clear()

        passDirlist =[]
        dirName = self.getProjectName()
        baseDir = "/show/" + dirName + "/shot/"
        allShotDir = os.listdir(baseDir)
        allShotDir.sort()
        for i in allShotDir:
            if i in passDirlist:
                continue
            if not i.istitle() and not i.startswith("."):
                pass
                self.listWidget.addItem(dirItem(i))







    def getProjectName(self):
        print(str(self.ui.searchLine.text()))
        return str(self.ui.searchLine.text())

    def makeBackdrop(self):
        selNodes = nuke.selectedNodes()
        if not selNodes:
            return nuke.nodes.BackdropNode()
        bdX = min([node.xpos() for node in selNodes])
        bdY = min([node.ypos() for node in selNodes])
        bdW = max([node.xpos() + node.screenWidth() for node in selNodes]) - bdX
        bdH = max([node.ypos() + node.screenHeight() for node in selNodes]) - bdY

        left, top, right, bottom = (-10, -80, 10, 10)
        bdX += left
        bdY += top - 50
        bdW += (right - left)
        bdH += (bottom - top) + 150
        n = nuke.nodes.BackdropNode(xpos=bdX,
                                    bdwidth=bdW,
                                    ypos=bdY,
                                    bdheight=bdH,
                                    tile_color=int((random.random() * (13 - 11))) + 11,
                                    note_font_size=42)
        n['selected'].setValue(False)
        for node in selNodes:
            node['selected'].setValue(True)
        return n

    def openSequence(self):


        xpos = 0
        ypos = 0
        for sequenceItem in self.listWidget.selectedItems():

            projectName = str(self.ui.searchLine.text())
            basePath = "/show/" + projectName + "/shot/" + sequenceItem.text()

            shotList = sorted(os.listdir(basePath))

            for shotName in shotList:
                print(shotName)
                shotPath = os.path.join(basePath, shotName)
                if (os.path.isdir(shotPath)) and ('_' in shotName):
                    platesPath = os.path.join(shotPath, 'plates')

                    if os.path.exists(platesPath):
                        platesDir = os.listdir(platesPath)
                        print(platesDir)
                        #
                        #                        if 'main' in platesDir:
                        #                            platesDir.insert(0, platesDir.pop(platesDir.index('main')))

                        for plateName in platesDir:
                            if 'retime' in plateName:
                                continue
                            elif plateName.startswith('.'):
                                continue

                            absPlatePath = os.path.join(platesPath, plateName)

                            # versionDir = sorted(os.listdir(absPlatePath), reverse=True)
                            versionDir = sorted(os.listdir(absPlatePath))
                            print('plate name : ', plateName)
                            print('version : ', versionDir)

                            if 'lo' in versionDir[0]:
                                versionDir.pop(0)
                            if 'png' in versionDir[0]:
                                versionDir.pop(0)
                            if 'jpg' in versionDir[0]:
                                versionDir.pop(0)
                            for pv in versionDir:
                                if pv.startswith('.'):
                                    continue
                                hvPath = absPlatePath + '/' + pv

                                files = QtCore.QDir(hvPath).entryList(QtCore.QDir.Files | QtCore.QDir.NoDotAndDotDot,
                                                                      QtCore.QDir.Name)

                                if files:
                                    rg = nuke.nodes.Read()
                                    files.sort()

                                    fileInfo = files[0].split('.')

                                    try:
                                        if '.dpx' in files[0]:
                                            paddingCount = len(files[0].split('.')[1])
                                        else:
                                           paddingCount = len(files[1].split('.')[1])
                                        # rg['file'].setValue(hvPath + '/'  + fileInfo[0] + '.####.dpx')
                                        rg['file'].setValue(hvPath + '/' + fileInfo[0] + '.' + '#' * paddingCount + '.dpx')

                                        rg['first'].setValue(int(fileInfo[1]))
                                        rg['last'].setValue(int(fileInfo[1]) + len(files) - 1)
                                        rg.setXYpos(xpos, ypos)
                                        ypos -= 100
                                    except:
                                        pass

                    nameNode = nuke.nodes.StickyNote()
                    nameNode['note_font_size'].setValue(20)
                    nameNode.setXYpos(xpos, 100)
                    nameNode['label'].setValue(shotName)
                    print("")
                    #                        dot = nuke.nodes.dot()
                    #                        dot.setXYpos(xpos, 0)
                    xpos += 200
                    ypos = 0
            ypos += 500

class dirItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent=None):
        super(dirItem, self).__init__(parent)

        self.itemFont = self.font()
        self.itemFont.setBold(True)
        self.itemFont.setPointSize(20)
        self.setFont(self.itemFont)
