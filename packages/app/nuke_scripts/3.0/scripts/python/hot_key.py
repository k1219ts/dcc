# -*- coding: utf-8 -*-
#Coded by hslth : lth1722@gmail.com

from PySide2 import QtWidgets, QtCore, QtGui

from ui_hotkey import Ui_Form
import sys, os, nuke, getpass, time

#reload(sys)
#sys.setdefaultencoding('utf-8')

class Hotkey(QtWidgets.QDialog):
    def __init__(self, parent):
        super(Hotkey, self).__init__(parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #self.basePath = '/mnt/3storage/Shared_Data/RnD_Team/hslth/Nuke_settings_hslth'
        #self.basePath = '/mnt/3storage/Shared_Data/Setting_Data/Nuke'
        #self.personalPath = '/python/personal_hotkey'
        self.treeModel = 0

        widgetPalette = self.palette()
        widgetPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(0,0,0))
        self.setPalette(widgetPalette)

        widgetFont = self.font()
        widgetFont.setBold(True)
        self.setFont(widgetFont)
        widgetFont.setPointSize(15)
        widgetPalette = self.ui.lineEdit_3.palette()
        widgetPalette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(255,165,0))

        self.ui.lineEdit_3.setFont(widgetFont)
        self.ui.lineEdit_3.setPalette(widgetPalette)

        self.ui.treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


        self.ui.textBrowser.setReadOnly(False)
        self.ui.textBrowser.setUndoRedoEnabled(False)

        self.hierarchy = ['', '', '', '', '','','']
        self.functionList = []
        self.ui.listWidget.setVisible(False)
        self.nukeCommandList = []
        self.nodeCommandList = []
        self.viewerCommandList = []
        self.commandMap = {'Nuke':self.nukeCommandList, 'Nodes':self.nodeCommandList, 'Viewer':self.viewerCommandList}
        self.modifierKey = {self.ui.pushButton_77:'Shift', self.ui.pushButton_78:'Ctrl', self.ui.pushButton_89:'Alt'}

        self.keyboardMap = {'Esc':self.ui.pushButton,'F1':self.ui.pushButton_2,'F2':self.ui.pushButton_3,
                            'F3':self.ui.pushButton_4,'F4':self.ui.pushButton_5,'F5':self.ui.pushButton_6,
                            'F6':self.ui.pushButton_7,'F7':self.ui.pushButton_8,'F8':self.ui.pushButton_9,
                            'F9':self.ui.pushButton_10,'F10':self.ui.pushButton_11,'F11':self.ui.pushButton_12,
                            'F12':self.ui.pushButton_13,'`':self.ui.pushButton_14, '~':self.ui.pushButton_14,
                            '1':self.ui.pushButton_15,'2':self.ui.pushButton_16,'3':self.ui.pushButton_17,
                            '4':self.ui.pushButton_18,'5':self.ui.pushButton_19,'6':self.ui.pushButton_20,
                            '7':self.ui.pushButton_21,'8':self.ui.pushButton_22,'9':self.ui.pushButton_23,
                            '0':self.ui.pushButton_24,'-':self.ui.pushButton_25,'=':self.ui.pushButton_26,
                            'Backspace':self.ui.pushButton_27,'Tab':self.ui.pushButton_28,'Q':self.ui.pushButton_29,
                            'W':self.ui.pushButton_30,'E':self.ui.pushButton_31,'R':self.ui.pushButton_32,
                            'T':self.ui.pushButton_33,'Y':self.ui.pushButton_34,'U':self.ui.pushButton_35,
                            'I':self.ui.pushButton_36,'O':self.ui.pushButton_37,'P':self.ui.pushButton_38,
                            '[':self.ui.pushButton_39,']':self.ui.pushButton_40,'\\':self.ui.pushButton_41,
                            'Caps':self.ui.pushButton_42,'A':self.ui.pushButton_43,'S':self.ui.pushButton_44,
                            'D':self.ui.pushButton_45,'F':self.ui.pushButton_46,'G':self.ui.pushButton_47,
                            'H':self.ui.pushButton_48,'J':self.ui.pushButton_49,'K':self.ui.pushButton_50,
                            'L':self.ui.pushButton_51,';':self.ui.pushButton_52,"'":self.ui.pushButton_53,
                            'Enter':self.ui.pushButton_54,'Shift':(self.ui.pushButton_55,self.ui.pushButton_66),
                            'Z':self.ui.pushButton_56,'X':self.ui.pushButton_57,'C':self.ui.pushButton_58,
                            'V':self.ui.pushButton_59,'B':self.ui.pushButton_60,'N':self.ui.pushButton_61,
                            'M':self.ui.pushButton_62,',':self.ui.pushButton_63,'.':self.ui.pushButton_64,
                            '/':self.ui.pushButton_65,'?':self.ui.pushButton_65,
                            'Ctrl':(self.ui.pushButton_67,self.ui.pushButton_76), 'Win':self.ui.pushButton_68,
                            'Alt':(self.ui.pushButton_69, self.ui.pushButton_73), u'한자':self.ui.pushButton_70,
                            'Space':self.ui.pushButton_71, u'한/영':self.ui.pushButton_72,
                            '↑':self.ui.pushButton_79,'←':self.ui.pushButton_80,
                            '↓':self.ui.pushButton_81,'→':self.ui.pushButton_82,
                            'Insert':self.ui.pushButton_83, 'Home':self.ui.pushButton_84,
                            'PageUp':self.ui.pushButton_85, 'Del':self.ui.pushButton_86,
                            'Insert':self.ui.pushButton_87, 'PageDown':self.ui.pushButton_88}

        self.mainGrayColor = self.ui.pushButton.palette().button().color().red()

        self.setCheckboxColor()
        self.setExceptionButton()
        self.setPropertySignal()
        self.startParsing()

        #=======================================================================
        # ETC, SIGNAL MAPPING
        #=======================================================================
        self.ui.lineEdit.textChanged.connect(self.searchTyped)
        self.ui.lineEdit_2.textChanged.connect(self.searchTyped_assign)
        self.ui.listWidget.itemDoubleClicked.connect(self.searchDoubleclicked)
        self.ui.treeWidget.itemDoubleClicked.connect(self.unasignedItem)
        self.ui.treeWidget.itemClicked.connect(self.unassignedStatus)

        self.ui.checkBox.stateChanged.connect(self.checkboxChanged)
        self.ui.checkBox_2.stateChanged.connect(self.checkboxChanged)
        self.ui.checkBox_3.stateChanged.connect(self.checkboxChanged)

        self.ui.pushButton_77.toggled.connect(self.setModifierText)
        self.ui.pushButton_78.toggled.connect(self.setModifierText)
        self.ui.pushButton_89.toggled.connect(self.setModifierText)

        self.ui.pushButton_90.clicked.connect(self.saveClicked)

    def getCheckedBoxlist(self):
        resultList = []
        for i in [self.ui.checkBox, self.ui.checkBox_2, self.ui.checkBox_3]:
            if i.checkState() == 2:
                resultList.append(i)
        return resultList

    def checkboxChanged(self, state):
        senderColor = self.sender().palette().base().color()

        for i in self.commandMap[str(self.sender().text())]:
            #print(i.text())
            if type(i) == tuple:
                for j in i:
                    if j.property(str(self.sender().text())):
                        if state == 2:
                            j.setStyleSheet('QPushButton{background-color: rgb(127,59,47);}')
                        else:
                            if self.getCheckedBoxlist():
                                for k in self.getCheckedBoxlist():
                                    if j.property(str(k.text())):
                                        pass
                                    else:
                                        #j.setStyleSheet('QPushButton{background-color: rgb(80.07,80.07,80.07);}')
                                        j.setStyleSheet('QPushButton{background-color: rgb(%s, %s, %s);}') %(self.mainGrayColor, self.mainGrayColor, self.mainGrayColor)
                            else:
                                #j.setStyleSheet('QPushButton{background-color: rgb(80.07,80.07,80.07);}')
                                j.setStyleSheet('QPushButton{background-color: rgb(%s, %s, %s);}') %(self.mainGrayColor, self.mainGrayColor, self.mainGrayColor)

            else:
                if i.property(str(self.sender().text())):
                    #print(i.palette().button().color().red())
                    if state == 2:

                        #if i.palette().button().color().red() == 80: # if button color is gray
                        #print(self.mainGrayColor)
                        if i.palette().button().color().red() == self.mainGrayColor: # if button color is gray

                            self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(0,0,0))
                            self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, senderColor)
                            i.setPalette(self.checkboxPalette)

                        else:
                            newColor = QtGui.QColor(0,0,0)
                            self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, newColor)
                            self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(120,120,120))

                            i.setPalette(self.checkboxPalette)

                    else:
                        if self.getCheckedBoxlist():
                            temp = []
                            for j in self.getCheckedBoxlist():
                                if i.property(str(j.text())):
                                    temp.append(j)
                                else:
                                    pass
                            if temp:
                                if len(temp)>1:
                                    newColor = QtGui.QColor(0,0,0)
                                    self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(120,120,120))

                                else:
                                    newColor = temp[0].palette().base().color()
                                    self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(0,0,0))

                                self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, newColor)
                                i.setPalette(self.checkboxPalette)
                                pass

                            else:
                                self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(0,0,0))
                                #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(80.07,80.07,80.07))
                                self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(self.mainGrayColor, self.mainGrayColor, self.mainGrayColor))

                                i.setPalette(self.checkboxPalette)
                        else:
                            #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(80.07,80.07,80.07))
                            self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(self.mainGrayColor, self.mainGrayColor, self.mainGrayColor))

                            i.setPalette(self.checkboxPalette)

    def getHierarchy(self, step, isColor):
        hierarchyStr = ''
        for i in range(step+1):
            if i == step:
                if isColor:
                    hierarchyStr = hierarchyStr + '<font size="5" color="DeepSkyBlue">' + self.hierarchy[i] + "</font>"
                else:
                    hierarchyStr = hierarchyStr + self.hierarchy[i] + '//'
            else:
                hierarchyStr = hierarchyStr + self.hierarchy[i] + '//'

        return hierarchyStr

    def getProperty(self):
        self.ui.textBrowser.clear()
        for i in self.getCheckedBoxlist():
            if self.sender().property(str(i.text())):
                self.ui.textBrowser.append(self.sender().property(str(i.text())))

    def searchDoubleclicked(self, item):
        self.ui.textBrowser.setText(item.toolTip())

    def searchTyped(self, typedStr):
        if typedStr:
            self.ui.listWidget.clear()
            for i in self.functionList:
                if typedStr.lower() in i[0].lower():
                    tempItem = QtWidgets.QListWidgetItem(i[0])
                    tempItem.setToolTip(i[1])
                    self.ui.listWidget.addItem(tempItem)

            self.ui.listWidget.setVisible(True)
        else:
            self.ui.listWidget.setVisible(False)

    def searchTyped_assign(self, typedStr):
        if self.ui.comboBox.currentIndex() == 0:
            self.ui.treeWidget.collapseAll()
            self.ui.treeWidget.clearSelection()
            if typedStr:
                searchedItem = self.ui.treeWidget.findItems(typedStr, QtCore.Qt.MatchRecursive|QtCore.Qt.MatchContains)
                if len(searchedItem)>1:
                    pass
                else:
                    self.ui.treeWidget.setCurrentItem(self.ui.treeWidget.headerItem())
                    pass

                for i in searchedItem[::-1]:
                    i.setSelected(True)
                    self.ui.treeWidget.setCurrentItem(i, 0, QtWidgets.QItemSelectionModel.NoUpdate)

    def startParsing(self):
        starttime = time.time()
        self.hot = nuke.hotkeys()
        for i in self.hot.split('\n'):
            step = int((len(i) - len(i.lstrip())-1)/4)
            self.hierarchy[step] = i.split('\t')[0].strip()

            if i.strip() == '' or i.strip() == '\t':
                continue
                #pass
            else:
                if step == 0:
                    tempItem = QtWidgets.QTreeWidgetItem(self.ui.treeWidget)
                    tempItem.setText(0, i.split('\t')[0].strip())
                    tempItem.setText(1, 'MENU')

                else:
                    if '\t' in str(i):
                        self.getParentItem(self.getHierarchy(step, False)[:-2], 'COMMAND', i.split('\t')[-1])
                        self.setButtonData(i, step)

                    else:
                        self.getParentItem(self.getHierarchy(step, False)[:-2], 'MENU', '')

        print(time.time() - starttime)

    def setButtonData(self, data, step):
        if (len(data.split('\t')) >1) and not(data.split('\t')[-1] == ''):
            self.functionList.append((data.split('\t')[0].strip(), '<font size="4" color="Black">' + self.getHierarchy(step, True)[:-2] + "</font><br/>" + '<font size=5 color="Orange">' + "Hotkey : " + data.split('\t')[-1] + "</font><br/><br/>"))
            key = data.split('\t')[-1].split('+')[-1]
            if key == "Left":
                key = "←"
            elif key == "Up":
                key = "↑"
            elif key == "Down":
                key = "↓"
            elif key == "Right":
                key = "→"

            button = None

            if self.keyboardMap.get(key):
                button = self.keyboardMap[key]
            if button is not None :
                if self.hierarchy and self.hierarchy[0]in self.commandMap:
                    if button in self.commandMap[self.hierarchy[0]]:
                        pass
                    else:
                        self.commandMap[self.hierarchy[0]].append(button)
            else:
                return
                
            if button == tuple:
                for j in button:
                    desc = j.property(self.hierarchy[0])
                    if desc == None:
                        desc = ''
                    j.setProperty(self.hierarchy[0], desc + '<font size="4" color="Black">' + self.getHierarchy(step, True)[:-2] + "</font><br/>" +
                                  '<font size=5 color="Orange">' + "Hotkey : " + data.split('\t')[-1] + "</font><br/><br/>")

                    j.setToolTip(str(button.toolTip()) + '<font size="4" color="Black">' + self.getHierarchy(step, True)[:-2] + "</font><br/>" +
                                  '<font size=5 color="Orange">' + "Hotkey : " + data.split('\t')[-1] + "</font><br/><br/>")

            elif button:
                desc = button.property(self.hierarchy[0])
                if desc == None:
                    desc = ''
                button.setProperty(self.hierarchy[0], desc + '<font size="4" color="Black">' + self.getHierarchy(step, True)[:-2] + "</font><br/>" +
                                  '<font size=5 color="Orange">' + "Hotkey : " + data.split('\t')[-1] + "</font><br/><br/>")
                button.setToolTip(str(button.toolTip()) + '<font size="4" color="Black">' + self.getHierarchy(step, True)[:-2] + "</font><br/>" +
                                  '<font size=5 color="Orange">' + "Hotkey : " + data.split('\t')[-1] + "</font><br/><br/>")


    def getParentItem(self, name, type, hotkey):
        tempParent = self.ui.treeWidget.findItems(name.split('//')[0],QtCore.Qt.MatchExactly)
        if tempParent and len(tempParent) == 1:
            self.searchNextitem(tempParent[0], name, 1, type, hotkey)

    def searchNextitem(self, parentItem, name, step, type, hotkey):
        tempBrush = QtGui.QBrush()
        if step > len(name.split('//'))-1:
            return

        else:
            tempItem = QtWidgets.QTreeWidgetItem()
            if parentItem.childCount() == 0:
                tempItem = QtWidgets.QTreeWidgetItem(parentItem)
                tempItem.setText(0, name.split('//')[step])
                tempItem.setText(1, type)

                if type == 'COMMAND':
                    tempBrush.setColor(QtGui.QColor(59,185,255))
                    tempItem.setForeground(1, tempBrush)
                    tempItem.setData(0, QtCore.Qt.UserRole, name)
                    tempBrush.setColor(QtGui.QColor(255,165,0))
                    tempItem.setForeground(2, tempBrush)
                    tempItem.setText(2, hotkey)
            else:
                found = False
                for i in range(parentItem.childCount()):
                    if parentItem.child(i).text(0) == name.split('//')[step]:
                        tempItem = parentItem.child(i)
                        found = True
                        break

                if found:
                    pass
                else:
                    tempItem = QtWidgets.QTreeWidgetItem(parentItem)
                    tempItem.setText(0, name.split('//')[step])
                    tempItem.setText(1, type)
                    if type == 'COMMAND':
                        tempBrush.setColor(QtGui.QColor(59,185,255))
                        tempItem.setForeground(1, tempBrush)
                        tempItem.setData(0, QtCore.Qt.UserRole, name)
                        tempBrush.setColor(QtGui.QColor(255,165,0))
                        tempItem.setForeground(2, tempBrush)
                        tempItem.setText(2, hotkey)

            self.searchNextitem(tempItem, name, step + 1, type, hotkey)

    def setPropertySignal(self):
        for j in self.keyboardMap.values():
            if type(j) == tuple:
                j[0].clicked.connect(self.getProperty)
                j[1].clicked.connect(self.getProperty)

            else:
                j.clicked.connect(self.getProperty)

    def setExceptionButton(self):
        """
        propertyText = '<font size="4" color="Black">Nuke/<font size="5" color="DeepSkyBlue">Delete</font></font><br/><font size=5 color="Orange">Hotkey : Delete</font><br/>'
        self.ui.pushButton_86.setProperty('Nuke', propertyText)
        self.ui.pushButton_86.setToolTip(str(self.ui.pushButton_86.toolTip()) + propertyText)
        self.nukeCommandList.append(self.ui.pushButton_86)
        """

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">First frame</font></font><br/><font size=5 color="Orange">Hotkey : Home</font>'
        self.ui.pushButton_84.setProperty('Viewer', propertyText)
        self.ui.pushButton_84.setToolTip(str(self.ui.pushButton_84.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_84)

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">Last frame</font></font><br/><font size=5 color="Orange">Hotkey : End</font>'
        self.ui.pushButton_87.setProperty('Viewer', propertyText)
        self.ui.pushButton_87.setToolTip(str(self.ui.pushButton_87.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_87)

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">Play backward</font></font><br/><font size=5 color="Orange">Hotkey : J</font><br/>'
        self.ui.pushButton_49.setProperty('Viewer', propertyText)
        self.ui.pushButton_49.setToolTip(str(self.ui.pushButton_49.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_49)

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">Stop</font></font><br/><font size=5 color="Orange">Hotkey : K</font><br/><br/>'
        self.ui.pushButton_50.setProperty('Viewer', propertyText)
        self.ui.pushButton_50.setToolTip(str(self.ui.pushButton_50.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_50)

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">Play forward</font></font><br/><font size=5 color="Orange">Hotkey : L</font><br/><br/>'
        self.ui.pushButton_51.setProperty('Viewer', propertyText)
        self.ui.pushButton_51.setToolTip(str(self.ui.pushButton_51.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_51)

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">Back 1 frame</font></font><br/><font size=5 color="Orange">Hotkey : Left</font><br/><br/>' + '<font size="4" color="Black">Viewer/<font size="5" color="DeepSkyBlue">Previous increment</font></font><br/><font size=5 color="Orange">Hotkey : Shift+Left</font><br/><br/>'
        self.ui.pushButton_80.setProperty('Viewer', propertyText)
        self.ui.pushButton_80.setToolTip(str(self.ui.pushButton_80.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_80)

        propertyText = '<font size="4" color="Black">Viewer//<font size="5" color="DeepSkyBlue">Forward 1 frame</font></font><br/><font size=5 color="Orange">Hotkey : Right</font><br/><br/>' + '<font size="4" color="Black">Viewer/<font size="5" color="DeepSkyBlue">Next increment</font></font><br/><font size=5 color="Orange">Hotkey : Shift+Right</font><br/><br/>'
        self.ui.pushButton_82.setProperty('Viewer', propertyText)
        self.ui.pushButton_82.setToolTip(str(self.ui.pushButton_82.toolTip()) + propertyText)
        self.viewerCommandList.append(self.ui.pushButton_82)

    def setCheckboxColor(self):
        self.ui.checkBox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ui.checkBox_2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ui.checkBox_3.setFocusPolicy(QtCore.Qt.NoFocus)

        self.checkboxPalette = QtGui.QPalette()
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(0,0,0))

        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(255,0,0))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(31,31,31))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(41,94,82))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25,140,130))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(84,30,50))
        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(81,115,3))
        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(0,0,0))


        self.ui.checkBox.setPalette(self.checkboxPalette)

        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(0,255,0))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(55,55,55))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(242,224,133))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(50,64,21))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(142,53,87))
        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(217,191,61))

        self.ui.checkBox_2.setPalette(self.checkboxPalette)

        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(0,0,255))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(141,141,141))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(238,127,56))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(140,55,39))
        #self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(136,163,62))
        self.checkboxPalette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(89,50,2))
        self.ui.checkBox_3.setPalette(self.checkboxPalette)

    def setModifierText(self, checked):
        if checked:
            if self.ui.lineEdit_3.text():
                temp = str(self.ui.lineEdit_3.text())
                if '+' in temp:
                    self.ui.lineEdit_3.setText(temp[:temp.rfind('+')+1] +
                                             self.modifierKey.get(self.sender()) + '+' + temp[temp.rfind('+')+1:])
                else:
                    self.ui.lineEdit_3.setText((self.modifierKey.get(self.sender()) + '+') + temp)
            else:
                self.ui.lineEdit_3.insert(self.modifierKey.get(self.sender()) + '+')
        else:
            self.ui.lineEdit_3.setText(str(self.ui.lineEdit_3.text()).replace(str(self.modifierKey.get(self.sender()) + '+'), ''))

    def unasignedItem(self, item, column):
        print(item.text(0))
        print(item.data(0, QtCore.Qt.UserRole))

        """
        print(item.data(0, QtCore.Qt.UserRole))
        ce = Asignkey(self, item.data(0, QtCore.Qt.UserRole))
        ok = ce.exec_()
        if ok:
            print("ok")
            #commandHotkey = self.setCommandHotkey(str(ce.ui.lineEdit.text()))
            commandHotkey = str(ce.ui.lineEdit.text())
            print(commandHotkey)
            self.assignHotkey(str(item.data(0, QtCore.Qt.UserRole)), commandHotkey)
        """
    def unassignedStatus(self, item, column):
        data = str(item.data(0, QtCore.Qt.UserRole))
        if item.data(0, QtCore.Qt.UserRole):
            self.ui.label_3.setText('<font size="5" color="DeepSkyBlue">' + item.data(0, QtCore.Qt.UserRole))
            #self.ui.label_3.setText('<font size="5" color="Black">' + data[:data.rfind('//')-2] + '<font size="5" color="DeepSkyBlue">' + data[data.rfind('//')+2:] + '</font>')
        else:
            self.ui.label_3.setText('')
        self.ui.lineEdit_3.setText(item.text(2))


    def saveClicked(self):
        pass
        """
        item = self.ui.treeWidget.currentItem()
        commandHotkey = str(self.ui.lineEdit_3.text()).replace(' ', '')
        self.assignHotkey(str(item.data(0, QtCore.Qt.UserRole)).replace('//', '/'), commandHotkey)
        """

    def assignHotkey(self, command, hotkey):
        print(command)
        """
        try:
            item = nuke.menu(command.split('/')[0]).findItem(command[command.find('/')+1:])
            #nuke.menu(command.split('/')[0]).addCommand(command[command.find('/')+1:],item.script(),hotkey)
            nuke.menu(command.split('/')[0]).menu(command[command.find('/')+1:]).setShortcut(hotkey)

            #===================================================================
            # TO DO : SAVE TO MENU.PY FILE REGARDING "getpass.getuser()"
            #self.basePath = '/home/hslth/nuke_local_setting/Nuke'
            #self.personalPath = '/python/personal_hotkey'
            #===================================================================
            descPath = self.basePath + self.personalPath + '/' + getpass.getuser()
            if os.path.exists(descPath):
                pass
            else:
                os.umask(0)
                os.makedirs(descPath, 0777)

            if os.path.exists(descPath + '/menu.py'):
                try:
                    f = file(descPath + '/menu.py', 'a')
                    #f.write("nuke.menu('%s').addCommand('%s','%s','%s')\n" %(command.split('/')[0],command[command.find('/')+1:],item.script(),hotkey))
                    f.write("nuke.menu('%s').menu('%s').setShortcut('%s')\n" %(command.split('/')[0],command[command.find('/')+1:],hotkey))
                    f.close()
                    self.ui.treeWidget.selectedItems()[0].setText(2, hotkey)
                except:
                     raise IOError("can't open or save personal hotkey file, ask Tae Hyung Lee!")

            else:
                try:
                    f = file(descPath + '/menu.py', 'w')
                    #f.write("nuke.menu('%s').addCommand('%s','%s','%s')\n" %(command.split('/')[0],command[command.find('/')+1:],item.script(),hotkey))
                    f.write("nuke.menu('%s').menu('%s').setShortcut('%s')\n" %(command.split('/')[0],command[command.find('/')+1:],hotkey))
                    f.close()
                    self.ui.treeWidget.selectedItems()[0].setText(2, hotkey)
                    os.umask(0)
                    os.chmod(descPath + '/menu.py', 0777)
                except:
                    raise IOError("can't open or save personal hotkey file, ask Tae Hyung Lee!")

            #nuke.menu(command.split('/')[0]).addCommand(command[command.find('/')+1:],item.script(),hotkey)
            nuke.message('<font size="5" color="White">Hotkey assigned.</font>\n' + '<font size="5" color="DeepSkyBlue">' + command + '</font>\n' + '<font size="5" color="Orange">Hotkey : ' + hotkey + '</font>')

        except:
            nuke.message('Error,,, Could be finditem error. Ask Tae Hyung Lee.')

        #nuke.menu("Nuke").addCommand("Edit/HSLTH/mergeLR", "mergeLR.getMergePath()","f10")
        """
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pd = Hotkey()
    pd.show()
    sys.exit(app.exec_())
