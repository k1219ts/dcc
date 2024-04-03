import hou,sys,ast,os
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class footprintpref(QtWidgets.QFrame):
    def __init__(self,parent=None):
        super(footprintpref, self).__init__(parent)
        
        self.setWindowTitle('Highlight Preferences')
        self.setFixedSize(400,0)
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)
       
        self.setProperty("houdiniStyle", True)
        self.setStyleSheet(hou.ui.qtStyleSheet())
        minsize = 75
       
        takegen = QtWidgets.QWidget()
        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        takegen.setLayout(layout)

        ## Materials

        groupBox = QGroupBox("Material Rings")
        layout.addWidget(groupBox)
        
        line1 = QVBoxLayout()
        line1.setSpacing(5)
        line1.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(line1)

        self.boxinfo = QLabel("When enabled selecting geo/material(SOP) nodes will\nhighlight material's associated with that node.\nHover over labels to see more information.")
        self.boxinfo.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        line1.addWidget(self.boxinfo)

        hor = QHBoxLayout()
        hor.setSpacing(5)
        hor.setSizeConstraint(QLayout.SetMinimumSize)
        line1.addLayout(hor)

        self.enable_mat_label = QLabel('Enable Flags:')
        self.enable_mat_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        hor.addWidget(self.enable_mat_label)

        self.enable_mat_button = QPushButton('Enable')
        self.enable_mat_button.setCheckable(1)
        hor.addWidget(self.enable_mat_button)

        hor = QHBoxLayout()
        hor.setSpacing(5)
        hor.setSizeConstraint(QLayout.SetMinimumSize)
        line1.addLayout(hor)

        self.deepdive_mat_label = QLabel('Search Children:')
        self.deepdive_mat_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.deepdive_mat_label.setToolTip('When enabled selections will look inside geo nodes for materials.\nSlower when enabled.')
        hor.addWidget(self.deepdive_mat_label)

        self.deepdive_mat_button = QPushButton('Enable')
        self.deepdive_mat_button.setCheckable(1)
        hor.addWidget(self.deepdive_mat_button)

        hor = QHBoxLayout()
        hor.setSpacing(5)
        hor.setSizeConstraint(QLayout.SetMinimumSize)
        line1.addLayout(hor)

        self.parm_mat_label = QLabel('Search Attributes:')
        self.parm_mat_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.parm_mat_label.setToolTip("When enabled selections will look for 'shop_materialpath' attributes\non SOPs with render/display flag enabled.\nSlower when enabled.")
        hor.addWidget(self.parm_mat_label)

        self.parm_mat_button = QPushButton('Enable')
        self.parm_mat_button.setCheckable(1)
        hor.addWidget(self.parm_mat_button)

        ## Cameras

        groupBox = QGroupBox("Camera Rings")
        layout.addWidget(groupBox)
        
        line1 = QVBoxLayout()
        line1.setSpacing(5)
        line1.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(line1)

        self.boxinfo = QLabel("When enabled changing the viewport camera\nwill highlight the camera in the network viewer.")
        self.boxinfo.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        line1.addWidget(self.boxinfo)

        hor = QHBoxLayout()
        hor.setSpacing(5)
        hor.setSizeConstraint(QLayout.SetMinimumSize)
        line1.addLayout(hor)

        self.enable_cam_label = QLabel('Enable Flags:')
        self.enable_cam_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        hor.addWidget(self.enable_cam_label)

        self.enable_cam_button = QPushButton('Enable')
        self.enable_cam_button.setCheckable(1)
        hor.addWidget(self.enable_cam_button)

        ## Lights

        groupBox = QGroupBox("Light Rings")
        layout.addWidget(groupBox)
        
        line1 = QVBoxLayout()
        line1.setSpacing(5)
        line1.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(line1)

        self.boxinfo = QLabel("When enabled, enabled lights\nwill highlight the camera in the network viewer")
        self.boxinfo.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        line1.addWidget(self.boxinfo)

        hor = QHBoxLayout()
        hor.setSpacing(5)
        hor.setSizeConstraint(QLayout.SetMinimumSize)
        line1.addLayout(hor)

        self.enable_lights_label = QLabel('Enable Flags:')
        self.enable_lights_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        hor.addWidget(self.enable_lights_label)

        self.enable_lights_button = QPushButton('Enable')
        self.enable_lights_button.setCheckable(1)
        hor.addWidget(self.enable_lights_button)

        ## Bottom strip

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(hou.qt.Separator())
        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        line1 = QHBoxLayout()
        line1.setSpacing(5)
        line1.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line1)

        self.info = QLabel('')
        line1.addWidget(self.info)

        line1.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.accept = QPushButton('Accept')
        self.accept.clicked.connect(self.saveChanges)
        line1.addWidget(self.accept)
        self.cancel = QPushButton('Close')
        self.cancel.clicked.connect(self.hide)
        line1.addWidget(self.cancel)

        self.enable_mat_button.setChecked(1)
        self.enable_cam_button.setChecked(1)
        self.enable_lights_button.setChecked(1)

        self.enable_mat_button.clicked.connect(self.clicked)
        self.deepdive_mat_button.clicked.connect(self.clicked)
        self.enable_cam_button.clicked.connect(self.clicked)

        self.pref = hou.expandString('$HOUDINI_USER_PREF_DIR/ring.pref')
        self.loadDefaults()

        self.setLayout(layout)

    def saveChanges(self):
        item = [self.enable_mat_button.isChecked(),
            self.deepdive_mat_button.isChecked(),
            self.enable_cam_button.isChecked(),
            self.parm_mat_button.isChecked(),
            self.enable_lights_button.isChecked()]

        a = file(self.pref, 'w')
        a.write(repr(item))

        self.info.setText('Preferences saved.\nHoudini restart required.')

    def loadDefaults(self):
        if os.path.isfile(self.pref):
            f = file(self.pref,'r')
            with open(self.pref, 'r') as f:
                recent = ast.literal_eval(f.read())

            try:
                self.enable_mat_button.setChecked(recent[0])
            except:
                pass
            try:
                self.deepdive_mat_button.setChecked(recent[1])
            except:
                pass
            try:
                self.enable_cam_button.setChecked(recent[2])
            except:
                pass
            try:
                self.parm_mat_button.setChecked(recent[3])
            except:
                pass
            try:
                self.enable_lights_button.setChecked(recent[4])
            except:
                pass

    def hideEvent(self, event):
        self.setParent(None)

    def clicked(self):
        self.info.clear()