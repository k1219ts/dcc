'''
v01
RuleBook Error로 임시로 설정해놓음. 추후 업데이트하며 룰북으로 수정할 것.
# import DXRulebook.Interface as rb
# import nukeCommon as comm
'''

import nuke
import os
import subprocess
from PySide2 import QtCore, QtGui, QtWidgets
from nukescripts import panels


class Opener_Genie(QtWidgets.QWidget):

    def __init__(self):

        super(Opener_Genie, self).__init__()

        self.setWindowTitle('Opener_Genie')
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        masterLayout = QtWidgets.QHBoxLayout()

        self.ScriptsOpenButton = QtWidgets.QPushButton('Scripts')
        self.ScriptsOpenButton.setMaximumWidth(100)
        self.ImagesOpenButton = QtWidgets.QPushButton('Images')
        self.ImagesOpenButton.setMaximumWidth(100)
        self.ScreeningOpenButton = QtWidgets.QPushButton('Screening')
        self.ScreeningOpenButton.setMaximumWidth(100)
        self.ShowOpenButton = QtWidgets.QPushButton('Show')
        self.ShowOpenButton.setMaximumWidth(100)


        ##### 경로설정 #####

        fullPath    = nuke.root().name()
        if fullPath == 'Root':
            scripts_path = None
            images_path = None
            screening_path = None
            show_path = None

        else :
            splitPath   = fullPath.split('/')
            show        = splitPath[2]
            seq         = splitPath[5]
            shot        = splitPath[6]
            comp        = splitPath[8]
            stat        = splitPath[-2]

            # Current Scripts
            scripts_path = fullPath
            # /images
            images_path = f"/show/{show}/_2d/shot/{seq}/{shot}/comp/{stat}/images/"
            # show폴더
            show_path = '/mach/show/'
            # 스크리닝경로 /show/<prj>/screening
            screening_path = show_path + show +'/screening/'


        # openButton
        self.ScriptsOpenButton.clicked.connect(lambda: self.open_folder(scripts_path))
        self.ImagesOpenButton.clicked.connect(lambda: self.open_folder(images_path))
        self.ScreeningOpenButton.clicked.connect(lambda: self.open_folder(screening_path))
        self.ShowOpenButton.clicked.connect(lambda: self.open_folder(show_path))

        masterLayout.addWidget(self.ScriptsOpenButton)
        masterLayout.addWidget(self.ImagesOpenButton)
        masterLayout.addWidget(self.ScreeningOpenButton)
        masterLayout.addWidget(self.ShowOpenButton)

        self.setLayout(masterLayout)

    def open_folder(self, folder_path):
        cmd = ['nautilus', '--no-desktop', '--browser', folder_path]
        if '8' == os.environ['REZ_CENTOS_MAJOR_VERSION']:
            cmd = ['nautilus', '--browser', folder_path]
        subprocess.Popen(cmd)


# Add Panel
panel = panels.registerWidgetAsPanel('Opener_Genie.Opener_Genie', 'Opener_Genie', 'Opener_Genie()', True)
