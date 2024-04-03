# import PySide
import mari
import os

# from pymodule.Qt import QtWidgets
from PySide2 import QtWidgets

# gui = PySide.QtGui

choise_types = ['all', 'selected']
exten_types = ['jpg', 'tif', 'bmp', 'org']

def showRefExportUI():

    global g_m2m_window
    g_m2m_window = QtWidgets.QDialog()
    m2m_layout = QtWidgets.QVBoxLayout()
    g_m2m_window.setLayout(m2m_layout)
    g_m2m_window.setWindowTitle("ref Export")

    #Create text
    step1_text = QtWidgets.QLabel("Output Folder")

    #Create browse layout
    browse_line_layout = QtWidgets.QHBoxLayout()
    global browse_line
    browse_line = QtWidgets.QLineEdit()
    browse_button = QtWidgets.QPushButton('Path')

    browse_line_layout.addWidget(browse_line)
    browse_line_layout.addWidget(browse_button)
    browse_button.clicked.connect(browseForFolder)

    #Add text to main layout
    m2m_layout.addWidget(step1_text)
    m2m_layout.addLayout(browse_line_layout)

    #Create layout bottom layout

    bottom_layout = QtWidgets.QHBoxLayout()

    #Create choise options
    choise_combo_text = QtWidgets.QLabel('all or selected')
    choise_combo = QtWidgets.QComboBox()
    for choise_type in choise_types:
        choise_combo.addItem(choise_type)
    choise_combo.setCurrentIndex(choise_combo.findText('all'))

    global exten_combo
    exten_combo_text = QtWidgets.QLabel('extension')
    exten_combo = QtWidgets.QComboBox()
    for exten_type in exten_types:
        exten_combo.addItem(exten_type)
    exten_combo.setCurrentIndex(exten_combo.findText('jpg'))

    bottom_layout.addWidget(choise_combo_text)
    bottom_layout.addWidget(choise_combo)
    bottom_layout.addWidget(exten_combo_text)
    bottom_layout.addWidget(exten_combo)

    #add bottom buttons layout, buttons and add

    main_ok_button = QtWidgets.QPushButton("OK")
    main_cancel_button = QtWidgets.QPushButton("Cancel")
    main_ok_button.clicked.connect(refImagExport)
    main_cancel_button.clicked.connect(g_m2m_window.reject)

    bottom_layout.addWidget(main_ok_button)
    bottom_layout.addWidget(main_cancel_button)

    m2m_layout.addLayout(bottom_layout)

    g_m2m_window.show()

def browseForFolder():
    #Get Folder
    dirname = str(QtWidgets.QFileDialog.getExistingDirectory(0,"Select Directory for Export"))

    if dirname:
        browse_line.setText(dirname)

def refImagExport():

    #get browse_line, exten_combo text value
    browse_line_text = browse_line.text
    exten_combo_fmt = exten_combo.currentText

    proj = mari.projects.current()

    a = str(proj)
    b = a.split('\'')
    projname = b[1]

    imgs = mari.images.list()

    for img in imgs:

        fuln = img.filePath().split('/')[-1]
        omtn = str(fuln)

        if "." in omtn:
            finn = omtn[:-4]
        else:
            finn = omtn

        img.saveAs(browse_line_text +'/' + projname + '_' + finn + '.' + exten_combo_fmt )
