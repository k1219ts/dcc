# -*- coding: utf-8 -*-
####################################################
#          coding by RND youkyoung.kim             #
####################################################
import os, sys, json
if sys.platform == 'win32':
    sys.path.append('N://backstage//pub//lib//python_lib')
import requests, dxConfig
import Qt
from Qt import QtWidgets
from Qt import QtCore
from camJsonWrite_ui import CameraSet_Form
import camJsonRead

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
CAMERA_MODEL = {'Alexa 4_3 ANA': [1.871, 0.702, 1.325],
                 'Red Dragon 6kFF': [1.209, 0.622, 10.95],
                 'Alexa SXT Plus': [1.111, 0.625, 1.000]}

if "Side" in Qt.__binding__:
    import maya.OpenMayaUI as mui
    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken
    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    def main():
        window = CameraWrite(getMayaWindow())
        window.move(QtWidgets.QDesktopWidget().availableGeometry().center() - window.frameGeometry().center())
        window.show()

if __name__ == "__main__":
    main()

class CameraWrite(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = CameraSet_Form()
        self.ui.setupUi(self)
        self.styleSetting()
        self.showInput()
        self.modelItems()
        self.connection()
        self.show()

    def connection(self):
        self.ui.model_combo.currentIndexChanged.connect(self.modelGet)
        self.ui.show_combo.currentIndexChanged.connect(self.currentShow)
        self.ui.help_btn.clicked.connect(self.helpModel)
        self.ui.ok_btn.clicked.connect(self.jsonWrite)
        self.ui.cancel_btn.clicked.connect(self.close)

    def styleSetting(self):
        self.ui.help_btn.setStyleSheet(
            "QPushButton#help_btn {background-color: #336699; border: 1px solid #999;}"
            "QPushButton#help_btn:hover {background-color: darkred; border: 1px solid #999;}"
        )

    def showInput(self):
        # show project list get and show combo additems
        self.projectDic = self.getProjectDict()
        projects = self.projectDic.keys()
        projects.sort()
        self.ui.show_combo.addItems(projects)
        self.currentShow()

    def currentShow(self):
        # select current show get
        title = self.ui.show_combo.currentText()
        self.curshow = self.projectDic[title]['name']

    def modelItems(self):
        # camera model combo additems
        self.camera_models = CAMERA_MODEL
        self.ui.model_combo.addItems(self.camera_models.keys())
        self.modelGet()

    def modelGet(self):
        # select current model get
        self.currentmodel = self.ui.model_combo.currentText()

    def jsonWrite(self):
        # camera json write
        jsondir, jsonfile = camJsonRead.jsonFileCheck(self.curshow)
        if jsondir and jsonfile:
            if not os.path.exists(jsondir):
                os.mkdir(jsondir)
            value = self.camera_models[self.currentmodel]
            jsondata = {'show': self.curshow,
                        'cameraModel': self.currentmodel,
                        'horizontalFilmAperture': value[0],
                        'verticalFilmAperture': value[1],
                        'overscan': value[2]}
            jsondata = json.dumps(jsondata, indent=4)

            if os.path.exists(jsonfile):
                title = "Warning : Current show json file existed !!"
                txt = "Overwrite save changes to \"%s\" camera json file?" % jsonfile
                ow = OpenWaringDialog(title, txt)
                ow.exec_()
                if ow.result == 'Save':
                    self.jsonOutput(jsonfile, jsondata)
                elif ow.result == 'Close':
                    ow.close()
            else:
                self.jsonOutput(jsonfile, jsondata)
                camJsonRead.messageBox('Sucess save layout.json file!!')

    def jsonOutput(self, jsonfile=None, jsondata=None):
        f = open(jsonfile, 'w')
        f.write(jsondata)
        f.close()

    def getProjectDict(self):
        # show project list get
        projectDic = {}
        params = {}
        params['api_key'] = API_KEY
        params['category'] = 'Active'
        infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()
        exceptList = ['test', 'testshot', 'china', 'vr']
        for i in infos:
            if i['name'] in exceptList:
                pass
            else:
                projectDic[i['title']] = i
        return projectDic

    def helpModel(self):
        # camera model list show
        helpwindow = CameraModel(QtWidgets.QDialog)
        helpwindow.show()
        result = helpwindow.exec_()

class CameraModel(QtWidgets.QDialog):
    # camera model dialog
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self)
        self.setWindowTitle("Camera Model")

        label = QtWidgets.QLabel("HFilmAperture, "
                                 "VFilmAperture, Overscan")
        commentBox = QtWidgets.QTextEdit()

        close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(close_btn)
        layout = QtWidgets.QGridLayout()

        layout.addWidget(label)
        layout.addWidget(commentBox)
        layout.addLayout(layout2,3,0)
        self.setLayout(layout)

        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)
        commentBox.setFont(font)
        model_sets = CAMERA_MODEL
        models = model_sets.keys()
        for i in model_sets:
            commentBox.append(i)
            commentBox.append('=' * 25)
            for j in model_sets[i]:
                commentBox.append(str(j))
            commentBox.append('\n')
        close_btn.clicked.connect(self.reject)

#----------------------------------------------------------
class OpenWaringDialog(QtWidgets.QMessageBox):
    # json file override ok, reject select
    def __init__(self, title=None, txt=None):
        QtWidgets.QMessageBox.__init__(self)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle(title)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText(txt)
        self.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Close)
        self.buttonClicked.connect(self.msgbtn)

    def msgbtn(self, i):
        self.result = i.text()
