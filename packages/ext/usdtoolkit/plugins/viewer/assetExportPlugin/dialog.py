from pxr.Usdviewq.qt import QtWidgets,QtCore

class GETTEXT():
    def __init__(self='',newName='',newShow='',element='',overwrite='',tag=''):
        self.newName = newName
        self.newShow = newShow
        self.element = element
        self.overwrite = overwrite
        self.tag = tag

def layout(dialog):
    (field, lineEditLayoutSet) = lineEditLayout(dialog)
    buttonLayout =button(dialog,field)

    vbox = QtWidgets.QVBoxLayout()
    vbox.addLayout(lineEditLayoutSet)
    vbox.addLayout(buttonLayout)
    return vbox

def button(dialog,field):
    pushButton_ok = QtWidgets.QPushButton("OK")
    pushButton_cancel = QtWidgets.QPushButton("Cancel")

    hbox = QtWidgets.QHBoxLayout()
    hbox.addWidget(pushButton_ok)
    hbox.addWidget(pushButton_cancel)

    pushButton_cancel.clicked.connect(lambda:dialog.close())
    pushButton_ok.clicked.connect(lambda:fillData(dialog,field).close())
    return hbox

def lineEditLayout(dialog):
    infoLabel = QtWidgets.QLabel("New Show: ")
    infoLabel2 = QtWidgets.QLabel("New AssetName: ")
    infoLabel3 = QtWidgets.QLabel("Element:")
    infoLabel4 = QtWidgets.QLabel("Overwrite:")
    infoLabel5 = QtWidgets.QLabel("Tag:")

    textLineEdit = QtWidgets.QLineEdit(dialog.data.newName)
    newShowLineEdit = QtWidgets.QLineEdit(dialog.data.newShow)
    elementCheckBox= QtWidgets.QCheckBox(dialog.data.element)
    overwriteCheckBox = QtWidgets.QCheckBox(dialog.data.overwrite)
    tagLineEdit = QtWidgets.QLineEdit(dialog.data.tag)

    hbox2 = QtWidgets.QHBoxLayout()
    hbox2.addWidget(infoLabel)
    hbox2.addWidget(newShowLineEdit)

    hbox3 = QtWidgets.QHBoxLayout()
    hbox3.addWidget(infoLabel2)
    hbox3.addWidget(textLineEdit)

    hbox4 = QtWidgets.QHBoxLayout()
    hbox4.addWidget(infoLabel3)
    hbox4.addWidget(elementCheckBox)
    hbox4.addWidget(infoLabel4)
    hbox4.addWidget(overwriteCheckBox)
    hbox4.addWidget(infoLabel5)
    hbox4.addWidget(tagLineEdit)

    vbox2 = QtWidgets.QVBoxLayout()
    vbox2.addLayout(hbox2)
    vbox2.addLayout(hbox3)
    vbox2.addLayout(hbox4)
    # buttonInfo =button(dialog, field).pushButton_cancel()

    return ({"newName": textLineEdit,
             "newShow": newShowLineEdit,
             "element":elementCheckBox,
             "overwrite":overwriteCheckBox,
             "tag": tagLineEdit},
            vbox2)

def fillData(dialog,field):
    dialog.data.newName = field["newName"].text()
    dialog.data.newShow = field["newShow"].text()
    dialog.data.tag = field["tag"].text()


    if field["element"].isChecked():
        dialog.data.element = True
    else:
        dialog.data.element = False

    if field["overwrite"].isChecked():
        dialog.data.overwrite = True
    else:
        dialog.data.overwrite = False

    return dialog


def dialogWindow(usdviewApi):
    window= usdviewApi.qMainWindow
    dialog = QtWidgets.QDialog(window)
    dialog.setWindowTitle("Input NewName")

    dialog.data = defaultInfo(usdviewApi,dialog)

    dialog.setMinimumWidth((window.size().width()/5))
    dialog.setLayout(layout(dialog))

    dialog.exec_()
    return (dialog.data.newShow,dialog.data.newName,dialog.data.element,dialog.data.overwrite,dialog.data.tag)

def defaultInfo(usdviewApi,dialog):
    newName = ''
    newShow = '/assetlib/3D'
    # newShow = '/show/pipe'
    element = ''
    overwrite =''
    tag=''
    return GETTEXT(newName,newShow,element,overwrite,tag)



