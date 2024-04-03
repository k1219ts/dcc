# -*- coding: utf-8 -*-
import getpass
import os
import sys
import time
import operator
import socket

if sys.platform == 'darwin':
    sys.path.insert(0, '/netapp/backstage/pub/lib/python_lib')

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

from ui_inventory import Ui_Form
from items import CategoryItem, ShowItem, Divider, TagItem, ThumbnailItem
import viewer
from user_config import GlobalConfig, ConfigThread

import db_query
import gc
import icon_rc
from dxstats import inc_tool_by_user

import dxConfig

def getParentWindow():
    try:
        import maya.OpenMayaUI as mui
        import shiboken2 as shiboken
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    except:
        try:
            import hou
            return hou.qt.mainWindow()
        except:
            return None

class Inventory(QtWidgets.QWidget):
    def __init__(self, parent=getParentWindow()):
        # UI setup
        print "UI SETUP"
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        inc_tool_by_user.run('inventory', getpass.getuser())
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle('Dexter Inventory By Tae Hyung Lee, Dexter RND')

        self.user_config = GlobalConfig.instance()

        self.ui.playSpeedSpin.setMinimum(0.01)

        self.ui.tagList.setViewMode(QtWidgets.QListView.ListMode)
        self.ui.tagList.setSortingEnabled(True)

        # DATE IN SEARCH TAB SETTING
        self.today = QtCore.QDate.currentDate()
        self.ui.toDate.setDate(self.today)
        self.ui.fromDate.setDate(self.today.addDays(-7))
        self.ui.fromDate.setEnabled(False)
        self.ui.toDate.setEnabled(False)
        self.enableDate(False)

        self.searchTimer = None
        self.configTimer = None

        self.tagContainer = []
        self.tagCount = db_query.getTagCount()

        self.__configUiSetting__()
        self.__connectSetting__()
        self.__shapeSetting__()
        self.__loadSetting__()
        # self.__frontPage__(type='FrontPage.Stats')

    def __configUiSetting__(self):
        configData = self.user_config.getConfig()
        if configData.has_key('play'):
            if configData['play'] == 'click':
                self.ui.clickToPlayButton.setChecked(True)
            else:
                self.ui.mouseOverToPlayButton.setChecked(True)
        if configData.has_key('auto_repeat'):
            self.ui.autoReplayCheck.setChecked(configData['auto_repeat'])
        if configData.has_key('item_per_click'):
            self.ui.itemPerClickSpin.setValue(configData['item_per_click'])
        if configData.has_key('play_speed'):
            self.ui.playSpeedSpin.setValue(configData['play_speed'])

        # BOOKMARK QUERY AND SHOW
        if configData.has_key('bookmark'):
            oids = configData['bookmark']
            self.ui.bookmarkList.objIds = oids
            for i in db_query.getByIds(oids):
                viewer.makeItems(self.ui.bookmarkList, i)

    def __connectSetting__(self):
        self.ui.categoryTree.itemDoubleClicked.connect(self.categoryDoubleClicked)

        self.ui.mainTab.tabCloseRequested.connect(self.closeTab)
        self.ui.menuTab.currentChanged.connect(self.menuTabChanged)

        # TWEAK TAGS
        self.ui.tagList.itemDoubleClicked.connect(self.tagListDoubleClicked)
        self.ui.tagLine.textChanged.connect(self.updateSearch)

        # RETURN PRESS & SEARCH BUTTON CLICK TO ACTUAL SEARCH
        self.ui.tagLine.returnPressed.connect(self.tagSearchPrepare)
        self.ui.searchButton.clicked.connect(self.tagSearchPrepare)

        # TAG SEARCH WITH DATE
        self.ui.uploadTimeCheck.stateChanged.connect(self.enableDate)

        self.ui.clickToPlayButton.clicked.connect(self.delay_for_config)
        self.ui.mouseOverToPlayButton.clicked.connect(self.delay_for_config)
        self.ui.autoReplayCheck.clicked.connect(self.delay_for_config)
        self.ui.itemPerClickSpin.valueChanged.connect(self.delay_for_config)
        self.ui.playSpeedSpin.valueChanged.connect(self.delay_for_config)

        self.ui.categoryCombo.activated.connect(self.refreshTag)

        self.ui.itemTagList.itemDoubleClicked.connect(self.itemTagDoubleClicked)

        # # actions short-cut
        self.ctrlW = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_W), self)
        self.ctrlW.activated.connect(self.shortCutCtrlW)

    def keyPressEvent(self, event):
        # TEST DEV FOR CATEGORY EDIT
        if event.key() == QtCore.Qt.Key_F5:
            print "refresh category"
            self.ui.categoryTree.clear()
            self.__loadSetting__()

        if event.key() == QtCore.Qt.Key_F8:
            print "edit category!!"
            item, ok = QtWidgets.QInputDialog.getText(self,
                                                      "Type Name of VFX Reference",
                                                      "Enter Name")
            if ok:
                db_query.insertOne({"type": "VFX_REF",
                                    "project": unicode(item),
                                    "enabled": False,
                                    })
                self.ui.categoryTree.clear()
                self.__loadSetting__()

    def __shapeSetting__(self):
        self.setStyleSheet("""
        Inventory{color: rgb(200,200,200); background: rgb(48,48,48); border-width: 1px;}""")

        self.ui.splitter.setStyleSheet("""
        QSplitter{color: rgb(200,200,200); background: rgb(48,48,48);}
        """)
        self.ui.splitter.setSizes([100, 1500, 250])

        self.ui.menuTab.setStyleSheet("""
        QWidget{color: rgb(200,200,200); background: rgb(48,48,48);}
        QTabWidget::pane{border: 1px solid #1f1f1f;}
        QTabWidget::tab-bar{alignment: left;}
        """)
        self.ui.mainTab.setUsesScrollButtons(True)
        self.ui.mainTab.setStyleSheet("""
        QTabWidget{color: rgb(200,200,200); background: rgb(48,48,48);}
        QTabWidget::pane{border: 1px solid #1f1f1f;}
        QTabWidget::tab-bar{alignment: left;}
        """)

        # Main tab widget for Category and Search
        self.ui.menuTab.tabBar().setStyleSheet("""
        QTabBar::tab{min-width: 90px; border: 1px solid #c4c4c4; color: rgb(200,200,200);
        border-color: rgb(33,33,33); border-bottom: none; border-top-left-radius: 4px;
        border-top-right-radius: 4px; padding: 4px;}
        QTabBar::tab:selected{background-color: rgb(48,48,48); margin-left: -4px; margin-right: -4px}
        QTabBar::tab:!selected{background-color: rgb(30,30,30); margin-top: 4px} 
        QTabBar::tab:first:selected{margin-left: 0} 
        QTabBar::tab:last:selected{margin-right: 0} 
        QTabBar::tab:only-one{margin: 0}
        """)

        # Sub tab widget for Listing thumbnail items
        self.ui.mainTab.tabBar().setStyleSheet("""
        QTabBar::tab{min-width: 150px; border: 1px solid #c4c4c4; border-color: rgb(33,33,33);
        color: rgb(200,200,200); border-bottom: none; border-top-left-radius: 4px;
        border-top-right-radius: 4px; padding: 4px}
        QTabBar::tab:selected{background-color: rgb(48,48,48); margin-left: -4px; margin-right: -4px} 
        QTabBar::tab:!selected{background-color: rgb(30,30,30); margin-top: 4px} 
        QTabBar::tab:first:selected{margin-left: 0} 
        QTabBar::tab:last:selected{margin-right: 0} 
        QTabBar::tab:only-one{margin: 0}
        """)

        self.ui.categoryTree.setIconSize(QtCore.QSize(30, 30))
        self.ui.categoryTree.setIndentation(10)
        self.ui.categoryTree.setStyleSheet("""
        QTreeView{border: none; color: rgb(200,200,200); background: rgb(48,48,48);}
        QTreeView::branch{background: rgb(48,48,48);}
        QTreeView::item:selected{background: rgb(83.895, 118.065, 153);}
        QTreeView::item:!selected{background: rgb(48,48,48);} 
        QTreeView::item:hover{color: rgb(0,0,0); background-color: rgb(247.095,146.88,30.09);}               
        QTreeView::branch:open:has-children:!has-siblings,
        QTreeView::branch:open:has-children:has-siblings  {border-image: none;image: url(:down-arrow.png);}
        QTreeView::branch:has-children:!has-siblings:closed,
        QTreeView::branch:closed:has-children:has-siblings {border-image: none; image: url(:right-arrow.png);}
        QTreeView::branch:closed:has-children {border-image: none; image: url(:right-arrow.png);}   
        """)

        self.ui.tagLine.setStyleSheet("""
        QLineEdit{border: 1px solid #1f1f1f;}
        QLineEdit:focus{border: 2px solid rgb(247.095,146.88,30.09);}
        """)

        self.ui.itemTagList.setStyleSheet("""
        QListView{border: none; background: rgb(48,48,48); color: rgb(200,200,200);}
        QListView::item:selected {background: rgb(68,150,210)}
        QListView::item:!selected {background: rgb(48,48,48)}
        QListView::item:hover{background-color: rgb(240,169,32);}
        """)

        scrollbarStyle = ''
        scrollbarStyle += 'QScrollBar{border: 1px solid black; background: #484848;} '
        scrollbarStyle += 'QScrollBar::add-page  {border: 1px solid grey; background: transparent;} '
        scrollbarStyle += 'QScrollBar::sub-page {border: 1px solid grey; background: transparent;} '
        self.ui.tagList.verticalScrollBar().setStyleSheet(scrollbarStyle)

        self.ui.categoryCombo.setStyleSheet("""
        QComboBox { color: white; padding: 0px 0px 0px 0px;}
        QComboBox QAbstractItemView
        {
            padding-top: 10px;
            padding-bottom: 10px;
            min-width: 150px;
            min-height: 90px;
            border: 1px solid gray;
        }
        """)
        self.ui.categoryCombo.setFocusPolicy(QtCore.Qt.NoFocus)

        self.ui.bookmarkLabel.setStyleSheet("""
        QLabel{border: none; background: rgb(48,48,48); color: rgb(200,200,200);}
        """)

    def __loadSetting__(self):
        waitRoot = CategoryItem(self.ui.categoryTree)
        waitRoot.setText(0, 'LOADING CATEGORY')

        waitRoot = CategoryItem(self.ui.categoryTree)
        waitRoot.setText(0, 'PLEASE WAIT')

        self.configThread = ConfigThread()
        self.configThread.emitMessage.connect(self.receiveTreeMessage)
        self.configThread.start()

    def receiveTreeMessage(self, message):
        print "MESSAGE RECEIVED!!!!"
        self.ui.categoryTree.clear()
        coreRoot = CategoryItem(self.ui.categoryTree)
        coreRoot.setText(0, 'CORE ELEMENT')
        coreRoot.setForeground(0, QtGui.QBrush(QtGui.QColor(247.095, 146.88, 30.09)))

        for coreType in message['CORE ELEMENT'].keys():
            coreItem = ShowItem(coreRoot)
            coreItem.setText(0, coreType)
            coreItem.searchTerm = message['CORE ELEMENT'][coreType]['searchTerm']

        coreRoot.setExpanded(True)

        del message['CORE ELEMENT']

        # # FRONT PAGE MENU
        frontPageItem = CategoryItem(self.ui.categoryTree)
        frontPageItem.setText(0, 'FRONT PAGE')
        frontPageItem.setForeground(0, QtGui.QBrush(QtGui.QColor(247.095, 146.88, 30.09)))

        frontPageSubItem = CategoryItem(frontPageItem)
        frontPageSubItem.setText(0, 'Upload Time')
        frontPageSubItem.setForeground(0, QtGui.QBrush(QtGui.QColor(83.895, 118.065, 153)))
        frontPageSubItem.id = 'FrontPage.Time'

        frontPageSubItem = CategoryItem(frontPageItem)
        frontPageSubItem.setText(0, 'Click Stats')
        frontPageSubItem.setForeground(0, QtGui.QBrush(QtGui.QColor(83.895, 118.065, 153)))
        frontPageSubItem.id = 'FrontPage.Stats'

        frontPageSubItem = CategoryItem(frontPageItem)
        frontPageSubItem.setText(0, 'Personal Stats')
        frontPageSubItem.setForeground(0, QtGui.QBrush(QtGui.QColor(83.895, 118.065, 153)))
        frontPageSubItem.id = 'FrontPage.Personal'

        frontPageSubItem = CategoryItem(frontPageItem)
        frontPageSubItem.setText(0, 'Recent Clicked Item')
        frontPageSubItem.setForeground(0, QtGui.QBrush(QtGui.QColor(83.895, 118.065, 153)))
        frontPageSubItem.id = 'FrontPage.recentClick'

        frontPageItem.setExpanded(True)

        # BASE CATEGORY
        for assetType in sorted(message.keys()):
            typeItem = CategoryItem(self.ui.categoryTree)
            typeItem.setText(0, assetType)
            typeItem.coll = assetType
            typeItem.searchTerm = message[assetType]['searchTerm']
            # print typeItem.searchTerm

            for prj in sorted(message[assetType].keys()):
                if prj == 'searchTerm':
                    continue
                prjItem = CategoryItem(typeItem)
                prjItem.setText(0, prj)
                prjItem.show = prj

                if not (message[assetType][prj].has_key('searchTerm')):
                    for cat in sorted(message[assetType][prj].keys()):
                        if cat == 'searchTerm':
                            continue
                        catItem = ShowItem(prjItem)
                        catItem.setText(0, cat)
                        catItem.searchTerm = {'type': assetType,
                                              'project': prj,
                                              'category': cat,
                                              'enabled': True}
                    pass
                else:
                    prjItem.id = 'Show'
                    prjItem.searchTerm = {'type': assetType,
                                          'project': prj,
                                          'enabled': True}

        self.configThread.wait()

    def __frontPage__(self, itemType):
        if itemType == 'FrontPage.Stats':
            tabContent = self.makeTabWithList('FRONT PAGE.Stat')

            diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
            diItem.setSizeHint(QtCore.QSize(1920, 50))
            diItem.setFlags(QtCore.Qt.ItemIsEnabled)
            divider = Divider()
            count = 20
            # divider.setItemName('Top %d Most <font size=10 color="Green">Clicked</font> Items' % count)
            divider.setItemName('Top %d Most <font size=10 color="#f0a920">Clicked</font> Items' % count)

            tabContent.listWidget.setItemWidget(diItem, divider)
            ids, docs = db_query.getStat(action='click', limit=count)

            for id in ids:
                for doc in docs:
                    if id == doc['_id']:
                        tabContent.insertSingleItem(tabContent.listWidget, doc)

            # for i in db_query.getStat(action='click', limit=count):
            #     tabContent.insertSingleItem(tabContent.listWidget, i)

            diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
            diItem.setSizeHint(QtCore.QSize(1920, 50))
            diItem.setFlags(QtCore.Qt.ItemIsEnabled)
            divider = Divider()
            # divider.setItemName('Top %d Most Double Clicked Items' % count)
            divider.setItemName('Top %d Most <font size=10 color="#f0a920">Double Clicked</font> Items' % count)
            tabContent.listWidget.setItemWidget(diItem, divider)

            # for i in db_query.getStat(action='doubleclick', limit=count):
            #     tabContent.insertSingleItem(tabContent.listWidget, i)
            ids, docs = db_query.getStat(action='doubleclick', limit=count)

            for id in ids:
                for doc in docs:
                    if id == doc['_id']:
                        print "found", doc['_id']
                        tabContent.insertSingleItem(tabContent.listWidget, doc)

            now = QtCore.QDateTime.currentDateTime()
            startDateTime = now.addDays(-30)

            toString = unicode(now.toString(QtCore.Qt.ISODate))
            fromString = unicode(startDateTime.toString(QtCore.Qt.ISODate))

            teamClickDic, teamClickData = db_query.team_click_count(startTime=fromString, endTime=toString)

            for team in sorted(teamClickDic):
                print "team", team
                if not (team in ['ani', 'fx', 'lnr', 'lay', 'cmp', 'mat', 'ast', 'fxfarm']):
                    continue
                clicked = sorted(teamClickDic[team]['click'].items(), reverse=True, key=operator.itemgetter(1))[:5]

                if clicked:
                    diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                    diItem.setSizeHint(QtCore.QSize(1920, 50))
                    diItem.setFlags(QtCore.Qt.ItemIsEnabled)
                    divider = Divider()
                    divider.setItemName(
                        'Top 5 Most <font size=10 color="#f0a920">%s</font> Item in <font size=10 color="Red">%s</font>' % (
                        'Clicked',
                        team.upper()))
                    tabContent.listWidget.setItemWidget(diItem, divider)
                    # print team, 'click'
                    for i in clicked:
                        # print i
                        for t in teamClickData:
                            if t['_id'] == i[0]:
                                tabContent.insertSingleItem(tabContent.listWidget, t)

                doubleclicked = sorted(teamClickDic[team]['doubleclick'].items(), reverse=True,
                                       key=operator.itemgetter(1))[
                                :5]
                if doubleclicked:
                    diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                    diItem.setSizeHint(QtCore.QSize(1920, 50))
                    diItem.setFlags(QtCore.Qt.ItemIsEnabled)
                    divider = Divider()
                    divider.setItemName(
                        'Top 5 Most <font size=10 color="#f0a920">%s</font> Item in <font size=10 color="Red">%s</font>' % (
                        'Double Clicked',
                        team.upper()))
                    tabContent.listWidget.setItemWidget(diItem, divider)
                    # print team, 'double click'
                    for i in doubleclicked:
                        # print i
                        for t in teamClickData:
                            if t['_id'] == i[0]:
                                tabContent.insertSingleItem(tabContent.listWidget, t)

        elif itemType == 'FrontPage.Time':
            newItems = db_query.getRecentUpdated()
            tabContent = self.makeTabWithList('FRONT PAGE.Time')

            for i in sorted(newItems):
                diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                diItem.setSizeHint(QtCore.QSize(1920, 50))
                # diItem.setFlags(QtCore.Qt.NoItemFlags)
                diItem.setFlags(QtCore.Qt.ItemIsEnabled)
                divider = Divider()
                divider.setItemName(i + ' Recent uploaded items')
                tabContent.listWidget.setItemWidget(diItem, divider)
                for item in newItems[i]:
                    tabContent.insertSingleItem(tabContent.listWidget, item)

        elif itemType == 'FrontPage.Personal':
            hostname = socket.gethostname().replace('.', '_')
            #hostname = 'lnr-eunok_kim'
            count = 10

            clickCount, doubleClickCount, clickLog, doubleClickLog = db_query.getPersonalClickStat(hostname,
                                                                                                   limit=count)

            tabContent = self.makeTabWithList('FRONT PAGE.Personal')
            if clickLog:
                diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                diItem.setSizeHint(QtCore.QSize(1920, 50))
                diItem.setFlags(QtCore.Qt.ItemIsEnabled)

                divider = Divider()
                divider.setItemName('<font size=10 color="#f0a920">%d Recent Clicked</font> Items' % count)

                tabContent.listWidget.setItemWidget(diItem, divider)

                for doc in reversed(clickLog):
                    tabContent.insertSingleItem(tabContent.listWidget, doc)

            if doubleClickLog:
                diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                diItem.setSizeHint(QtCore.QSize(1920, 50))
                diItem.setFlags(QtCore.Qt.ItemIsEnabled)
                divider = Divider()
                divider.setItemName('<font size=10 color="#f0a920">%d Recent Double Clicked</font> Items' % count)
                tabContent.listWidget.setItemWidget(diItem, divider)

                for doc in reversed(doubleClickLog):
                    tabContent.insertSingleItem(tabContent.listWidget, doc)

            if clickCount:
                diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                diItem.setSizeHint(QtCore.QSize(1920, 50))
                diItem.setFlags(QtCore.Qt.ItemIsEnabled)
                divider = Divider()
                divider.setItemName('<font size=10 color="#f0a920">%d Most Clicked</font> Items' % count)
                tabContent.listWidget.setItemWidget(diItem, divider)

                for doc in clickCount:
                    tabContent.insertSingleItem(tabContent.listWidget, doc)

            if doubleClickCount:
                diItem = QtWidgets.QListWidgetItem(tabContent.listWidget)
                diItem.setSizeHint(QtCore.QSize(1920, 50))
                diItem.setFlags(QtCore.Qt.ItemIsEnabled)
                divider = Divider()
                divider.setItemName('<font size=10 color="#f0a920">%d Most Double Clicked</font> Items' % count)
                tabContent.listWidget.setItemWidget(diItem, divider)

                for doc in doubleClickCount:
                    tabContent.insertSingleItem(tabContent.listWidget, doc)

        elif itemType == 'FrontPage.recentClick':
            tabContent = self.makeTabWithList('FRONT PAGE.recentClick')
            #configData = self.user_config.getConfig()
            #infos = db_query.getClickHistory(configData['item_per_click'])
            infos = db_query.getClickHistory(limit=200)

            for doc in infos:
                imgItem = viewer.makeItems(tabContent.listWidget, doc)
                if type(imgItem) == ThumbnailItem:
                    tabContent.listWidget.itemWidget(imgItem).titleLabel.setText(doc['user'])
                else:
                    imgItem.setText(doc['user'])

    def shortCutCtrlW(self):
        if self.ui.mainTab.currentIndex() >= 0:
            self.closeTab(self.ui.mainTab.currentIndex())

    def delay_for_config(self):
        if self.configTimer is not None:
            self.configTimer.stop()
            self.configTimer.start(300)
        else:
            self.configTimer = QtCore.QTimer()
            self.configTimer.setSingleShot(True)
            self.configTimer.timeout.connect(self.config_setting)
            self.configTimer.start(300)

    def config_setting(self):
        config_dic = {'user': getpass.getuser()}
        if self.ui.clickToPlayButton.isChecked():
            config_dic['play'] = 'click'
        else:
            config_dic['play'] = 'over'
        config_dic['auto_repeat'] = self.ui.autoReplayCheck.isChecked()
        config_dic['item_per_click'] = self.ui.itemPerClickSpin.value()
        config_dic['play_speed'] = self.ui.playSpeedSpin.value()

        self.user_config.update(config_dic)

    def closeTab(self, index):
        tabContent = self.ui.mainTab.widget(index)
        listwidget = tabContent.listWidget
        for i in range(listwidget.count()):
            listwidget.removeItemWidget(listwidget.item(i))

        listwidget.clear()
        listwidget.deleteLater()
        del (listwidget)
        del (tabContent)

        self.ui.mainTab.removeTab(index)

        gc.collect()

    def menuTabChanged(self, index):
        self.tagContainer = []

        if index == 1:
            self.tagContainer = db_query.getTagList()
            self.populateTags()

            self.ui.categoryCombo.clear()
            self.ui.categoryCombo.addItem('ALL')
            for i in db_query.getDistinct('type'):
                self.ui.categoryCombo.addItem(i)

    def enableDate(self, toggle):
        self.ui.toDate.setEnabled(toggle)
        self.ui.fromDate.setEnabled(toggle)
        if toggle:
            # CASE ENABLE STYLESHEET
            color = 'lightgray'

        else:
            # CASE ENABLE STYLESHEET
            color = 'black'
        css = """
        QDateEdit {border: 1px solid %s; color: %s;}
        QDateEdit::drop-down {border-left: 1px solid %s;}
        QDateEdit::down-arrow {image: url(:/down-arrow.png);
        width: 7px; height: 7px;}
        """ % (color, color, color)
        self.ui.toDate.setStyleSheet(css)
        self.ui.fromDate.setStyleSheet(css)

    def refreshTag(self, index):
        assetType = unicode(self.ui.categoryCombo.currentText())
        if assetType == 'ALL':
            self.tagContainer = db_query.getTagList()
        else:
            self.tagContainer = db_query.getFindDistinct({'type': assetType}, 'tags')
        self.populateTags()

    def categoryDoubleClicked(self, item, col):
        if not (item.id == 'Show'):
            if unicode(item.id).startswith('FrontPage'):
                print "call front page!!"
                self.__frontPage__(item.id)
            return

        tabContent = self.makeTabWithList(item.text(0))
        # print item.searchTerm
        result = db_query.searchByTerm(item.searchTerm)
        tabContent.setRecord(list(result))
        tabContent.insertThumbnailItem()

    def tagListDoubleClicked(self, item):
        tag = unicode(item.text())
        curLineText = unicode(self.ui.tagLine.text())
        if ' ' in curLineText:
            text = ' '.join(curLineText.split(' ')[:-1] + [tag])
            self.ui.tagLine.setText(text + ' ')
        else:
            self.ui.tagLine.setText(tag + ' ')

    def tagSearchPrepare(self):
        tag = unicode(self.ui.tagLine.text())
        if self.ui.matchAllRadio.isChecked():
            # SEARCH && MATCH
            tagTerm = ""
            for i in tag.split(' '):
                tagTerm += "\"" + i + '\" '
            tagTerm = tagTerm.strip()
        else:
            # SEARCH || MATCH
            tagTerm = tag

        optionTerm = {}
        # TIME OPTION
        if self.ui.uploadTimeCheck.isChecked():
            fromString = unicode(self.ui.fromDate.date().toString(QtCore.Qt.ISODate))
            toString = unicode(self.ui.toDate.date().toString(QtCore.Qt.ISODate))
            optionTerm = {'time': {'$gte': fromString + 'T00:00:00',
                                   '$lte': toString + 'T23:59:59'}}

        if not (self.ui.categoryCombo.currentText() == 'ALL'):
            optionTerm['type'] = unicode(self.ui.categoryCombo.currentText())

        if tagTerm == u'""':
            if optionTerm:
                print optionTerm
                # SEARCH ONLY BY TIME
                tabContent = self.makeTabWithList('search/by date')
                result = db_query.searchByTerm(optionTerm)

                tabContent.setRecord(list(result))
                tabContent.insertThumbnailItem()
            # NO INPUT
            return
        self.searchByTag(tagTerm, optionTerm)

    def searchByTag(self, tag, term=None):
        # TAG SHOULD BE STRING OR UNICODE
        title = tag.replace('"', '').replace("'", '')

        tabContent = self.makeTabWithList('search/' + title)

        result = db_query.searchByTag(tag, term)
        tabContent.setRecord(list(result))
        tabContent.insertThumbnailItem()

    def updateSearch(self, text):
        if self.searchTimer is not None:
            self.searchTimer.stop()
            self.searchTimer.start(200)
        else:
            self.searchTimer = QtCore.QTimer()
            self.searchTimer.setSingleShot(True)
            self.searchTimer.timeout.connect(self.populateTags)
            self.searchTimer.start(200)

    def populateTags(self):
        self.ui.tagList.clear()
        text = unicode(self.ui.tagLine.text())
        if ' ' in text:
            searchText = text.split(' ')[-1]
        else:
            searchText = text

        if searchText:
            self.ui.tagList.clear()
            searchedTag = [i for i in self.tagContainer if searchText.lower() in i.lower()]
            for tag in searchedTag:
                if tag:
                    item = QtWidgets.QListWidgetItem(self.ui.tagList)
                    item.setText(tag)
        else:
            for tag in self.tagContainer:
                if tag:
                    item = QtWidgets.QListWidgetItem(self.ui.tagList)
                    item.setText(tag)

    def makeTabWithList(self, title):
        # tabContent = TabWidget()
        tabContent = viewer.TabWidget()
        tabContent.listWidget.assetClicked.connect(self.assetClicked)

        self.ui.mainTab.addTab(tabContent, title)
        self.ui.mainTab.setCurrentWidget(tabContent)
        return tabContent

    def assetClicked(self, items):
        self.ui.itemTagList.clear()
        tags = []
        for i in items:
            tags += i.getItemData()['tags']

        tags = sorted(list(set(tags)))
        for i in range(len(tags)):
            if self.tagCount.has_key(tags[i]):
                item = TagItem(self.ui.itemTagList, self.tagCount[tags[i]])
            else:
                item = TagItem(self.ui.itemTagList)
            item.setText(tags[i])

    def itemTagDoubleClicked(self, item):
        # print item.text()
        tabContent = self.makeTabWithList('tag/' + item.text())
        searchTerm = {'enabled': True, '$text': {'$caseSensitive': False,
                                                 '$search': unicode(item.text())
                                                 }
                      }

        result = db_query.searchByTerm(searchTerm)
        tabContent.setRecord(list(result))
        tabContent.insertThumbnailItem()


def mountCheck():
    mount = True
    mountText = ''
    if sys.platform in ['darwin', 'linux2']:

        if os.getenv("SITE") == "KOR" and not(os.path.exists('/stdrepo')): # 10.0.0.14:/data2
            mount = False
            mountText += '10.0.0.228 Fail\n'
        if not (os.path.exists(dxConfig.getConf("ASSETLIB_PATH"))):  # 10.0.0.243
            mount = False
            mountText += '10.0.0.131/others Fail\n'
    else:
        # CASE WINDOWS??
        pass
    return mount, mountText


def main():
    print "MAIN!!"
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(":appIcon.png"))
    app.setDoubleClickInterval(200)
    spMode = True
    # spMode = False
    if spMode:

        splash = QtWidgets.QSplashScreen()
        splash.setWindowFlags(QtCore.Qt.SplashScreen | QtCore.Qt.WindowStaysOnTopHint)
        splash.show()

        count = 0
        limit = 44
        thInst = Inventory(None)
        start = time.time()
        while time.time() - start < 1:
            if count > limit:
                count = 0
            # print count
            imagename = ':splash_seq/imges.%s.png' % str(count + 1).zfill(4)
            # print imagename
            pixmap = QtGui.QPixmap(imagename)
            # splash.setMask(pixmap.mask())
            splash.setPixmap(pixmap)
            time.sleep(0.02)
            app.processEvents()
            count += 1
        app.processEvents()
        splash.finish(thInst)

    else:
        thInst = Inventory(None)

    # thInst = Inventory(None)
    # MOUNT CHECK
    mountResult = mountCheck()
    if not (mountResult[0]):
        QtWidgets.QMessageBox.information(thInst,
                                          u"마운트를 체크 하세요.",
                                          mountResult[1]
                                          )
    else:
        thInst.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
