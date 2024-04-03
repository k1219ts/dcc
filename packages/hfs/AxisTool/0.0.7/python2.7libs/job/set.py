import hou,os,ast,re,sys
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class jobSet(QtWidgets.QFrame):
    def __init__(self,parent=None):
        super(jobSet, self).__init__(parent)
        
        self.setProperty("houdiniStyle", True)
        self._error_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        self._error_brush.setStyle(QtCore.Qt.SolidPattern)

        self._warn_brush = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        self._warn_brush.setStyle(QtCore.Qt.SolidPattern)

        self._info_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        self._info_brush.setStyle(QtCore.Qt.SolidPattern)

        self._msg_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        self._msg_brush.setStyle(QtCore.Qt.SolidPattern)
        
        self.setWindowTitle('$JOB Set')
        self.resize(450,150)
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)
       
        sh = hou.ui.qtStyleSheet()
        self.setStyleSheet(sh)

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)
            
        minsize = 150

        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
       

        line = QHBoxLayout()    
        line.setSpacing(5)
        line.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line)

        self.check1 = QRadioButton()
        self.check1.setChecked(1)
        self.check1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.check1.toggled.connect(self.modeChange)
        line.addWidget(self.check1)
        self.label1 = QLabel('Folders Back From $HIP')
        self.label1.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        self.label1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.label1.setMinimumWidth(minsize)
        line.addWidget(self.label1)

        self.back = QSpinBox()
        self.back.setMinimum(0)
        self.back.setMinimumWidth(200)
        self.back.valueChanged.connect(self.setMode)
        line.addWidget(self.back)


        line = QHBoxLayout()    
        line.setSpacing(5)
        line.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line)

        self.check2 = QRadioButton()
        self.check2.toggled.connect(self.modeChange)
        line.addWidget(self.check2)
        self.label2 = QLabel('Manually Set Path')
        self.label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        self.label2.setMinimumWidth(minsize)
        line.addWidget(self.label2)

        self.path = QLineEdit()
        self.path.setPlaceholderText('JOB Path')
        self.path.textChanged.connect(self.setMode)
        line.addWidget(self.path)

        # path
        def onFileSelected(file_path):
            if file_path.endswith('/'): file_path = file_path[:-1]
            self.path.setText(hou.expandString(file_path))

        self.icon = hou.qt.FileChooserButton()
        self.icon.setFileChooserTitle("$JOB Path")
        self.icon.setFileChooserMode(hou.fileChooserMode.Write)
        self.icon.fileSelected.connect(onFileSelected)
        self.icon.setFileChooserFilter(hou.fileType.Directory)
        line.addWidget(self.icon)

        layout.addWidget(hou.qt.Separator())

        line = QHBoxLayout()    
        line.setSpacing(5)
        line.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line)

        self.info = QLabel()
        self.info.setEnabled(0)
        line.addWidget(self.info)
        line.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.execute = QPushButton('Set $JOB')
        self.execute.clicked.connect(self.changeJob)
        self.execute.setToolTip('Ctrl+Enter')
        line.addWidget(self.execute)

        self.setLayout(layout)

        shortcut1 = QShortcut(QKeySequence("Ctrl+Return"), self)
        #shortcut1.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        shortcut1.activated.connect(self.changeJob)

        self.pref = '%s/jobwidget.pref'%hou.expandString('$HOUDINI_USER_PREF_DIR').replace('\\','/')

        self.modeChange()
        self.setMode()
        self.loadDefaults()

    def setMode(self):
        if self.check1.isChecked():
            hipPath = hou.expandString('$HIP').replace('\\','/')
            self.jobPath = hipPath.rsplit('/',self.back.value())[0]
        else:
            self.jobPath = self.path.text()

        self.info.setText('\"%s\"'%self.jobPath)


    def modeChange(self):
        if self.check1.isChecked():
            self.label2.setEnabled(0)
            self.path.setEnabled(0)
            self.icon.setEnabled(0)
            self.label1.setEnabled(1)
            self.back.setEnabled(1)
        else:
            self.label2.setEnabled(1)
            self.path.setEnabled(1)
            self.icon.setEnabled(1)
            self.label1.setEnabled(0)
            self.back.setEnabled(0)

        self.setMode()

    def changeJob(self):
        if self.jobPath and os.path.isdir(self.jobPath):
            hou.hscript('setenv JOB = %s'%self.jobPath)

            self.info.setText('$JOB successfully changed.')
            self.saveChanges()
        else:
            hou.ui.displayMessage('Folder doesn\'t exist.', buttons=('OK',),default_choice=0, close_choice=0, title='Info')

    def saveChanges(self):
        item = [self.check1.isChecked(),self.back.value(),self.check2.isChecked(),self.path.text()]
        a = file(self.pref, 'w')
        a.write(repr(item))

    def loadDefaults(self):
        if os.path.isfile(self.pref):
            f = file(self.pref,'r')
            with open(self.pref, 'r') as f:
                recent = ast.literal_eval(f.read())

            self.check1.setChecked(recent[0])
            self.back.setValue(recent[1])
            self.check2.setChecked(recent[2])
            self.path.setText(recent[3])