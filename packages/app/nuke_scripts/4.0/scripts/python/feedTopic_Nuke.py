# -*- coding: utf-8 -*-

from PySide2 import QtWidgets, QtCore, QtGui

import sys, requests, nuke, os, math, subprocess
from tactic_client_lib import TacticServerStub
from dd_config import GlobalConfig

#from ui_feedTopic_Nuke import Ui_Dialog
import ui_feedTopic_Nuke
# reload(ui_feedTopic_Nuke)

class FeedTopic_Nuke(QtWidgets.QWidget):
    def __init__(self, parent):
        #UI setup
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_feedTopic_Nuke.Ui_Dialog()

        self.ui.setupUi(self)
        self.resize(1200, 648)

        self.widgetFont = self.font()
        self.widgetFont.setPointSize(11)
        self.setFont(self.widgetFont)

        #------------------------------------------------------------------------------
        self.configData = GlobalConfig.instance()
        self.prjDic = self.configData.prjConfigData['prj_code']
        self.prjNameDic = self.configData.prjConfigData['prj_name']
        self.seqDic = self.configData.prjConfigData['prj_seq']
        #------------------------------------------------------------------------------
        self.ui.prjCombo.addItem('None', None)

        for i in sorted(self.prjDic.keys(), reverse=True):
            if self.prjDic[i] in self.configData.prjConfigData['valid_prj']:
                #print(i, self.prjDic[i])
                #self.ui.prjCombo.addItem(i, self.prjDic[i])
                self.ui.prjCombo.addItem(i + '_' + self.prjDic[i], self.prjDic[i])

        self.ui.teamCombo.addItem('comp')
        self.ui.teamCombo.addItem('lighting')
        self.ui.teamCombo.addItem('fx')
        self.ui.teamCombo.addItem('publish')
        self.ui.teamCombo.addItem('Latest')
        #------------------------------------------------------------------------------
        #self.ui.snapshotTree.setSortingEnabled(True)
        self.ui.snapshotTree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ui.snapshotTree.setRootIsDecorated(False)
        self.ui.snapshotTree.setColumnCount(7)
        self.ui.snapshotTree.headerItem().setText(0, "Order")
        self.ui.snapshotTree.headerItem().setText(1, "Shot")
        self.ui.snapshotTree.headerItem().setText(2, "Status")
        self.ui.snapshotTree.headerItem().setText(3, "Process")
        self.ui.snapshotTree.headerItem().setText(4, "Context")
        self.ui.snapshotTree.headerItem().setText(5, "Frame_in")
        self.ui.snapshotTree.headerItem().setText(6, "Frame_out")
        self.ui.snapshotTree.header().resizeSection(0, 60)
        self.ui.snapshotTree.header().resizeSection(1, 120)
        self.ui.snapshotTree.header().resizeSection(2, 120)
        self.ui.snapshotTree.header().resizeSection(4, 150)
        #------------------------------------------------------------------------------
        self.ui.loopRadio.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ui.holdRadio.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ui.noneRadio.setFocusPolicy(QtCore.Qt.NoFocus)
        self.checkboxPalette = QtGui.QPalette()

        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(81,115,3))
        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(0,0,0))
        self.ui.loopRadio.setPalette(self.checkboxPalette)

        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(217,191,61))
        self.ui.holdRadio.setPalette(self.checkboxPalette)

        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(89,50,2))
        self.ui.noneRadio.setPalette(self.checkboxPalette)
        self.ui.loopRadio.setChecked(True)
        #------------------------------------------------------------------------------
        self.ui.topicList.itemClicked.connect(self.updateItemList)
        self.ui.snapshotTree.itemDoubleClicked.connect(self.playItem)
        self.ui.importButton.clicked.connect(self.importItems)

        self.ui.pdplayerButton.clicked.connect(self.playPdplayer)

        self.ui.closeButton.clicked.connect(self.closeWidget)

        self.ui.prjCombo.currentIndexChanged.connect(self.updateTopic)

        self.ui.teamCombo.currentIndexChanged.connect(self.updateProcess)

        #------------------------------------------------------------------------------
        self.HOST = '10.0.0.51'
        self.API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


    def getCurrentPrjcode(self):
        prj = str(self.ui.prjCombo.itemData(self.ui.prjCombo.currentIndex()))
        prjCode = self.prjNameDic[prj]
        return prjCode

    def updateTopic(self, index):
        self.ui.topicList.clear()
        self.ui.snapshotTree.clear()
        if unicode(self.ui.prjCombo.currentText()) == 'None':
            pass
        else:
            prjCode = self.getCurrentPrjcode()
            for topic in self.getJson(self.HOST, 'topic', {'project_code':prjCode}):
                item = TopicItem(self.ui.topicList, topic)
                item.setText(item.topicData['title'])

    def updateProcess(self, index):
        if self.ui.topicList.selectedItems():
            self.updateItemList(self.ui.topicList.selectedItems()[0])


    def updateItemList(self, item):
        self.ui.snapshotTree.clear()
        prjCode = self.getCurrentPrjcode()
        process = self.ui.teamCombo.currentText()
        context = '%s/%s' % (process, process)

        if process in ['Latest', 'All']:
            items = self.getJson(self.HOST, 'submission_topic',
                                 {'project_code':prjCode,'topic_code':item.topicData['code']})

            for shot in items:
                if shot['process']:
                    item = TopicShotItem(self.ui.snapshotTree, shot)
                    item.setText(0, str(shot['sort_order']))
                    item.setText(1, shot['code'])
                    item.setText(2, shot['status'])
                    item.setText(3, shot['process'])
                    item.setText(4, shot['context'])
                    item.setText(5, str(shot['frame_in']))
                    item.setText(6, str(shot['frame_out']))
                    try:
                        item.setForeground(2, self.configData.colorScheme[shot['status']])
                        item.setForeground(3, self.configData.teamColor[shot['process']])
                    except:

                        item.setForeground(2, QtGui.QColor('#ffffff'))
                        item.setForeground(3, QtGui.QColor('#ffffff'))
        else:
            items = self.getJson(self.HOST, 'submission_topic',
                                 {'project_code':prjCode,'topic_code':item.topicData['code'],
                                  'process':process})

            for shot in items:
                #print(shot['code'], process, shot['context'])
                #if shot['process'] == process:
                #if shot['context'].startswith(context):
                if process in shot['context'].split('/')[-1]:

                    item = TopicShotItem(self.ui.snapshotTree, shot)
                    item.setText(0, str(shot['sort_order']))
                    item.setText(1, shot['code'])
                    item.setText(2, shot['status'])
                    item.setText(3, shot['process'])
                    item.setText(4, shot['context'])
                    item.setText(5, str(shot['frame_in']))
                    item.setText(6, str(shot['frame_out']))
                    try:
                        item.setForeground(2, self.configData.colorScheme[shot['status']])
                        item.setForeground(3, self.configData.teamColor[shot['process']])
                    except:
                        item.setForeground(2, QtGui.QColor('#ffffff'))
                        item.setForeground(3, QtGui.QColor('#ffffff'))
                else:
                    item = TopicShotItem(self.ui.snapshotTree, shot)
                    item.setText(0, str(shot['sort_order']))
                    item.setText(1, shot['code'])
                    item.setText(2, shot['status'])
                    item.setText(3, shot['process'])
                    item.setText(4, shot['context'])
                    item.setText(5, str(shot['frame_in']))
                    item.setText(6, str(shot['frame_out']))
                    try:
                        item.setForeground(2, self.configData.colorScheme[shot['status']])
                    except:
                        item.setForeground(2, QtGui.QColor('#ffffff'))
                    try:
                        item.setForeground(3, self.configData.teamColor[shot['process']])
                    except:
                        item.setForeground(3, QtGui.QColor('#ffffff'))


    def createRead(self, shotName, filePath, start, end):
#        dur = end -start + 1
#        rNode = nuke.createNode('Read', inpanel=False)

        rNode = nuke.createNode('Read', 'file "%s"' % filePath,inpanel=False)
        dur = rNode['first'].value() - rNode['last'].value()
        refNode = nuke.nodes.Reformat()
        refNode['type'].setValue('to format')
        refNode['format'].setValue('HD_1080')
        refNode.setInput(0, rNode)
#        rNode['file'].setValue(filePath)
#        rNode['first'].setValue(1)
#        rNode['origfirst'].setValue(1)
#        rNode['last'].setValue(dur)
#        rNode['origlast'].setValue(dur)

        tNode = nuke.nodes.Text2()
        tNode['message'].setValue(shotName)
        tNode.setInput(0, refNode)
        tNode['box'].setX(0)
        tNode['box'].setY(rNode.height())
        tNode['box'].setR(500.0)
        tNode['box'].setT(rNode.height()* 0.95)
        tNode['label'].setValue("[value message]")

        if self.ui.holdRadio.isChecked():
            fh = nuke.nodes.FrameHold()
            fh['first_frame'].setValue(dur / 2)
            fh.setInput(0, tNode)
            #fh.setYpos(tNode.ypos() + 70)

        elif self.ui.loopRadio.isChecked():
            fh = nuke.nodes.Retime()
            fh['before'].setValue('loop')
            fh['after'].setValue('loop')
            fh['label'].setValue("[value input.last]")
            fh.setInput(0, tNode)
            #fh.setYpos(tNode.ypos() + 70)
            # forceValidate doesn't works. show and hide
            nuke.show(fh)
            fh.hideControlPanel()

        else:
            fh = tNode

        return (rNode, tNode, fh)

    def importItems(self):
        fhCount = 0
        if not(self.ui.snapshotTree.selectedItems()):
            self.ui.snapshotTree.selectAll()

        cs = nuke.nodes.ContactSheet()
        basePos = None
        readList = []

        #for index in range(self.ui.snapshotTree.topLevelItemCount()):
#        for index, item in enumerate(self.ui.snapshotTree.selectedItems()):
#            item = self.ui.snapshotTree.topLevelItem(index)
        for item in self.ui.snapshotTree.selectedItems():
            if item.shotData['status'] == 'Omit':
                continue
            dur = item.shotData['frame_out'] - item.shotData['frame_in'] + 1

            if os.path.isfile(item.shotData['tactic_path']):
                rNode, tNode, fh = self.createRead(item.shotData['code'],
                                                   item.shotData['tactic_path'],
                                                   item.shotData['frame_in'],
                                                   item.shotData['frame_out'])
                cs.setInput(fhCount, fh)
                if basePos:
                    rNode.setXYpos(basePos[0] + 100, basePos[1])
                    tNode.setXYpos(basePos[0] + 100, basePos[1]+100)
                    if fh == rNode:
                        print("no option")
                        pass
                    else:
                        fh.setXYpos(basePos[0] + 100, basePos[1]+200)

                else:
                    basePos = [rNode.xpos(), rNode.ypos()]
                    tNode.setXYpos(rNode.xpos(), rNode.ypos()+100)
                    if fh == rNode:
                        print("no option")
                        pass
                    else:
                        fh.setXYpos(basePos[0], basePos[1]+200)
                readList.append(rNode)
                basePos = [rNode.xpos(), rNode.ypos()]
                fhCount += 1

            else:
                uploadRoot = str(item.shotData['tactic_path'])
                for i in os.listdir(uploadRoot):
                    rNode, tNode, fh = self.createRead(item.shotData['code'],
                                                       uploadRoot + '/' + i,
                                                       item.shotData['frame_in'],
                                                       item.shotData['frame_out'])
                    cs.setInput(fhCount, fh)
                    if basePos:
                        rNode.setXYpos(basePos[0] + 100, basePos[1])
                        tNode.setXYpos(basePos[0] + 100, basePos[1]+100)
                        if fh == rNode:
                            pass
                        else:
                            fh.setXYpos(basePos[0] + 100, basePos[1]+200)
                    else:
                        basePos = [rNode.xpos(), rNode.ypos()]
                        tNode.setXYpos(rNode.xpos(), rNode.ypos()+100)
                        if fh == rNode:
                            pass
                        else:
                            fh.setXYpos(basePos[0], basePos[1]+200)
                    readList.append(rNode)
                    basePos = [rNode.xpos(), rNode.ypos()]
                    fhCount += 1

        cs.setXYpos(((readList[-1].xpos() - readList[0].xpos())/2) + readList[0].xpos(),
                    readList[0].ypos() + 500)

        row = math.sqrt(len(readList))
        if row.is_integer():
            column = row
        else:
            if (row - int(row)) > 0.5:
                row = row + 1
                column = row
            else:
                column = row + 1

        cs['rows'].setValue(int(row))
        cs['columns'].setValue(int(column))
        cs['width'].setValue(1920*int(column))
        cs['height'].setValue(1080*int(row))
        cs['roworder'].setValue('TopBottom')


    def getJson(self, host, searchType, params):
        params['api_key'] = self.API_KEY
        return requests.get("http://%s/dexter/search/%s.php" % (host, searchType),params = params).json()


    def playItem(self, item, column):
        #Double click event handling
        videoPath = item.shotData['tactic_path']
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(videoPath, QtCore.QUrl.TolerantMode))

    def playPdplayer(self):
        exe = 'pdplayer'
        movList = [exe]
        if not(self.ui.snapshotTree.selectedItems()):
            self.ui.snapshotTree.selectAll()

        for index, item in enumerate(self.ui.snapshotTree.selectedItems()):
            item = self.ui.snapshotTree.topLevelItem(index)
            if os.path.isfile(item.shotData['tactic_path']):
                movList.append(item.shotData['tactic_path'])

            else:
                uploadRoot = str(item.shotData['tactic_path'])
                for i in os.listdir(uploadRoot):
                    movList.append(uploadRoot + '/' + i)

        subprocess.Popen(movList)

    def closeWidget(self):
        self.close()

class TopicItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent=None, data=None):
        super(TopicItem, self).__init__(parent)
        self.topicData = data
#        if data:
#            for attr in data.keys():
#                setattr(self, attr, data[attr])


class TopicShotItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, data=None):
        super(TopicShotItem, self).__init__(parent)
        self.shotData = data
#        for i in range(self.columnCount()):
#            self.setTextAlignment(i, QtCore.Qt.AlignCenter)
        for i in range(7):
            self.setTextAlignment(i, QtCore.Qt.AlignCenter)



def main():
    app = QtWidgets.QApplication(sys.argv)
    fd = FeedTopic_Nuke(None)
    fd.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
