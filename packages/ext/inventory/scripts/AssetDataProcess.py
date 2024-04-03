#####################
# date      : 2017.09.11
# author    : daeseok.chae by DEXTER RND
# use tool  : maya
#
#####################
import os
from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import time
import string
import subprocess

scriptRoot = os.path.dirname(os.path.abspath(__file__))
if '/home/' in scriptRoot:
    # scriptRoot = '/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/scripts/AssetBackupTool'
    scriptRoot = '/dexter/Cache_DATA/RND/daeseok/Inventory'

def openDirBtnClicked(lineEditWidget, showComboBox, typeComboBox, assetNameLineEdit):
    print "dirOpen", lineEditWidget.text()
    dirPath = lineEditWidget.text()

    if os.path.isdir(dirPath) == False:
        dirPath = os.path.dirname(dirPath)

    while(os.path.exists(dirPath) == False):
        dirPath = os.path.dirname(dirPath)
        if dirPath == '/show':
            break

    dialog = QtWidgets.QFileDialog()
    dialog.setMinimumSize(800, 400)
    dialog.setFileMode(QtWidgets.QFileDialog.Directory)
    dialog.setDirectory(dirPath)

    result = dialog.exec_()
    if result == 1:
        print dialog.selectedFiles()
        filePath = dialog.selectedFiles()[-1]

        if filePath.startswith('/netapp/dexter/show'):
            filePath.replace('/netapp/dexter/show', '/show')

        lineEditWidget.setText(filePath)

        # try:
        splitText = str(filePath).split('/')
        print "openDirSplitText ", splitText

        project = splitText[2]
        showComboBox.setCurrentIndex(showComboBox.findText(project))

        typeName = splitText[4]
        typeComboBox.setCurrentIndex(typeComboBox.findText(typeName))

        assetName = splitText[5]
        assetNameLineEdit.setText(assetName)
        # except Exception as e:
        #     print e.message

def updatePreviewPath(type, updateText, lineEditWidget):
    print "updatePreview", updateText, lineEditWidget.text()
    splitText = lineEditWidget.text().split('/')

    print splitText

    path = "/show"
    try:
        if type == "show":
            path = os.path.join(path, updateText)
        else:
            path = os.path.join(path, splitText[2])

        path = os.path.join(path, 'asset')

        if type == "type":
            path = os.path.join(path, updateText)
        else:
            path = os.path.join(path, splitText[4])

        if type == "name":
            if updateText != "":
                path = os.path.join(path, updateText, "model/pub")
            else:
                path = os.path.join(path, splitText[5], "model/pub")
        else:
            path = os.path.join(path, splitText[5], "model/pub")

        if type == "zenv":
            if updateText:
                path = os.path.join(path, 'zenv', 'abc')
            else:
                path = os.path.join(path, 'scenes')
        else:
            if len(splitText) > 8:
                print splitText[8:]
                path = os.path.join(path, string.join(splitText[8:], '/'))
            else:
                path = os.path.join(path, 'scenes')

    except:
        pass

    finally:
        lineEditWidget.setText(path)

def exportAssetData(parent, itemDic):
    dialog = QtWidgets.QDialog(parent)
    dialog.setGeometry(0, 0, 730, 230)
    dialog.setWindowTitle("Export Item")

    filesDic = itemDic['files']

    dialog.setStyleSheet('''QDialog{color: rgb(200,200,200); background: rgb(48,48,48); border-width: 1px;}
                            QLineEdit{border: 1px solid rgb(200, 200, 200);  background: rgb(48, 48, 48); color: rgb(200, 200, 200); }
                            QLineEdit:focus{border: 2px solid rgb(200, 200, 200); }
                            QLabel{border: none; background: rgb(48,48,48); color: rgb(200,200,200);}
                            QCheckBox {border: none; background: rgb(48,48,48); color: rgb(200,200,200);}
                            QCheckBox:disabled { color: rgb(80, 80, 80); }
                            QComboBox { background: rgb(48, 48, 48); color: rgb(200,200,200); padding: 0px 0px 0px 0px;
                                        border: 1px solid rgb(200, 200, 200); }
                            QComboBox::drop-down { border: 2px solid rgb(200, 200, 200); }
                            QComboBox:focus{ border:2px solid rgb(200, 200, 200); }
                            QComboBox QAbstractItemView { background: #191B1E; padding-top: 10px; padding-bottom: 10px; 
                                                            min-width: 150px; min-height: 90px; 
                                                            border: 1px solid rgb(200, 200, 200); }
                            QPushButton { background: #2A2A2A; color: rgb(200, 200, 200); padding: 0px 0px 0px 0px;}
                            ''')

    label1 = QtWidgets.QLabel(dialog)
    label1.setText("show")
    label1.setFont(QtGui.QFont("Cantarell", 14))
    label1.setGeometry(10, 10, 60, 30)

    showComboBox = QtWidgets.QComboBox(dialog)
    showComboBox.setFont(QtGui.QFont("Cantarell", 14))
    showComboBox.setGeometry(70, 10, 130, 30)

    label2 = QtWidgets.QLabel(dialog)
    label2.setText("type")
    label2.setFont(QtGui.QFont("Cantarell", 14))
    label2.setGeometry(210, 10, 60, 30)

    typeComboBox = QtWidgets.QComboBox(dialog)
    typeComboBox.setFont(QtGui.QFont("Cantarell", 14))
    typeComboBox.setGeometry(260, 10, 130, 30)

    label3 = QtWidgets.QLabel(dialog)
    label3.setText("assetName")
    label3.setFont(QtGui.QFont("Cantarell", 14))
    label3.setGeometry(400, 10, 130, 30)

    assetNameLineEdit = QtWidgets.QLineEdit(dialog)
    assetNameLineEdit.setGeometry(510, 10, 120, 30)
    assetNameLineEdit.setFont(QtGui.QFont("Cantarell", 14))

    isZenvCheckbox = QtWidgets.QCheckBox(dialog)
    isZenvCheckbox.setGeometry(640, 10, 80, 30)
    isZenvCheckbox.setText("Zenv?")

    label4 = QtWidgets.QLabel(dialog)
    label4.setText("export model path")
    label4.setFont(QtGui.QFont("Cantarell", 14))
    label4.setGeometry(10, 50, 200, 30)

    previewModelPath = QtWidgets.QLineEdit(dialog)
    previewModelPath.setGeometry(180, 50, 500, 30)
    previewModelPath.setText("/show")
    previewModelPath.setFont(QtGui.QFont("Cantarell", 14))
    previewModelPath.setReadOnly(True)

    openDirBtn = QtWidgets.QPushButton(dialog)
    openDirBtn.setGeometry(690, 50, 30, 30)
    imagePath_folder = '/dexter/Cache_DATA/RND/jeongmin/CacheExport_jm/resources/folder.png'
    openDirBtn.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))
    openDirBtn.clicked.connect(lambda : openDirBtnClicked(previewModelPath, showComboBox, typeComboBox, assetNameLineEdit))

    showComboBox.currentIndexChanged.connect(lambda : updatePreviewPath("show", showComboBox.currentText(), previewModelPath))
    typeComboBox.currentIndexChanged.connect(lambda: updatePreviewPath("type", typeComboBox.currentText(), previewModelPath))
    assetNameLineEdit.textChanged.connect(lambda: updatePreviewPath("name", assetNameLineEdit.text(), previewModelPath))
    isZenvCheckbox.clicked.connect(lambda: updatePreviewPath("zenv", isZenvCheckbox.checkState() == QtCore.Qt.Checked, previewModelPath))

    showComboBox.addItems(os.listdir('/show'))
    typeComboBox.addItems(['char', 'env', 'prop', 'vehicle'])

    checkBoxNameList = ['model', 'low model', 'model json',
                        'tex files', 'image files', 'proxy files',
                        'shader', 'binding', 'mari', 'hair', 'zbrush']

    hasKeyList = {'model' : 0,
                  'model_low' : 1,
                  'model_json' : 2,
                  'tex' : 3,
                  'images' : 4,
                  'proxy' : 5,
                  'shader_json' : 6,
                  'shader_xml' : 7,
                  'mari' : 8,
                  'hair' : 9,
                  'zbrush' : 10}

    reverseHasKeyList = {0: 'model',
                  1: 'model_low',
                  2: 'model_json',
                  3: 'tex',
                  4: 'images',
                  5: 'proxy',
                  6: 'shader_json',
                  7: 'shader_xml',
                  8: 'mari',
                  9: 'hair',
                  10: 'zbrush'}

    checkBoxList = []
    print "Files :", filesDic.keys()
    for index, value in enumerate(checkBoxNameList):
        i = index / 5
        j = index % 5
        checkBoxList.append(QtWidgets.QCheckBox(dialog))
        checkBoxList[index].setGeometry(10 + 120 * j, 90 + 30 * i, 80, 30)
        checkBoxList[index].setText(value)
        checkBoxList[index].setChecked(True)

        print index, reverseHasKeyList[index]
        if not reverseHasKeyList[index] in filesDic.keys():
            checkBoxList[index].setDisabled(True)
            checkBoxList[index].setChecked(False)

    okButton = QtWidgets.QPushButton(dialog)
    okButton.setText("OK")
    okButton.clicked.connect(dialog.accept)
    okButton.setGeometry(270, 190, 60, 30)

    closeButton = QtWidgets.QPushButton(dialog)
    closeButton.setText("Cancel")
    closeButton.clicked.connect(dialog.reject)
    closeButton.setGeometry(340, 190, 60, 30)

    retDlg = dialog.exec_()
    print "DialogExec :", retDlg
    print QtWidgets.QDialog.Accepted
    print QtWidgets.QDialog.Rejected

    if retDlg == QtWidgets.QDialog.Rejected:
        return

    # Key Type
    #  - model   : model, model_low, model_json
    #  - zbrush  : zbrush
    #  - texture : mari, tex, images
    #  - mari    : mari
    #  - shader  : shader_xml, shader_json
    #  - hair    : hair

    cmdList = []

    _env = SetupEnvironment(2017)

    # splitText = previewModelPath.text().split('/')
    #
    # project = splitText[2]
    # assetPath = splitText[3:5]
    #
    # assetName = splitText[5]
    # originalAssetName = string.join(os.path.basename(filesDic['model']).split('_')[:-1], '_')
    # layerName = originalAssetName
    # beehive_model_GRP(original) => test_model_GRP(export)
    # layerName_model_GRP

    # if layer name changed,
    # 1. model
    #  - layer name attr change
    #  - model node name change
    #  - model export file name change
    # 2. texture
    #  - tex file name change
    #  - images file name change
    # 3. shader
    #  - shader node name change
    #  - shader export file name change
    #  - if shader file is exist, append?
    #  - shader binding rules name change

    # if assetName == originalAssetName:
    #     layerName = originalAssetName
    # else:
    #     layerName = assetName

    # model

    if isZenvCheckbox.checkState() == QtCore.Qt.Checked:
        modelPath = os.path.join(previewModelPath.text(), itemDic['name'], 'model')
    else:
        modelPath = os.path.join(previewModelPath.text())

    if checkBoxList[hasKeyList['model']].checkState() == QtCore.Qt.Checked \
        or checkBoxList[hasKeyList['model_low']].checkState() == QtCore.Qt.Checked \
        or checkBoxList[hasKeyList['model_json']].checkState() == QtCore.Qt.Checked:

        if  not os.path.exists(modelPath):
            cmd = "install -d -m 755 {0}".format(modelPath)
            cmdList.append(cmd)

        if checkBoxList[hasKeyList['model']].checkState() == QtCore.Qt.Checked and filesDic.has_key('model'):
            # cmd = "cp -rf {0} {1}".format(filesDic['model'], modelPath)
            assetNameAttr = string.join(modelPath.split('/')[2:5], '/')

            if isZenvCheckbox.checkState() == QtCore.Qt.Checked:
                exportPath = os.path.join(modelPath, os.path.basename(filesDic['model']).replace('.abc', '_v01.abc'))
            else:
                exportPath = os.path.join(modelPath, '%s_model_v01.abc' % assetNameLineEdit.text())

            cmd = '/usr/autodesk/maya2017/bin/mayapy %s/Asset/exportModelBatch.py %s %s %s' % (scriptRoot,
                                                                                               filesDic['model'],
                                                                                               assetNameAttr,
                                                                                               exportPath)

            cmdList.append(cmd)

        if checkBoxList[hasKeyList['model_low']].checkState() == QtCore.Qt.Checked and filesDic.has_key('model_low'):
            # cmd = "cp -rf {0} {1}".format(filesDic['model_low'], modelPath)
            assetNameAttr = string.join(modelPath.split('/')[2:5], '/')

            if isZenvCheckbox.checkState() == QtCore.Qt.Checked:
                exportPath = os.path.join(modelPath, os.path.basename(filesDic['model_low']).replace('.abc', '_v01.abc'))
            else:
                exportPath = os.path.join(modelPath, '%s_model_v01.abc' % assetNameLineEdit.text())

            cmd = '/usr/autodesk/maya2017/bin/mayapy %s/Asset/exportModelBatch.py %s %s %s' % (
            scriptRoot, filesDic['model_low'], assetNameAttr, exportPath)

            cmdList.append(cmd)

        if checkBoxList[hasKeyList['model_json']].checkState() == QtCore.Qt.Checked and filesDic.has_key('model_json'):
            exportPath = os.path.join(modelPath, os.path.basename(filesDic['model_json']).replace('.json', '_v01.json'))
            print exportPath
            cmd = "cp -rf {0} {1}".format(filesDic['model_json'], exportPath)
            cmdList.append(cmd)

    # tex, image, proxy
    if checkBoxList[hasKeyList['tex']].checkState() == QtCore.Qt.Checked \
        or checkBoxList[hasKeyList['images']].checkState() == QtCore.Qt.Checked \
        or checkBoxList[hasKeyList['proxy']].checkState() == QtCore.Qt.Checked:

        texturePath = os.path.join(modelPath.split('/model')[0], 'texture/pub/tex/v01')
        if checkBoxList[hasKeyList['tex']].checkState() == QtCore.Qt.Checked and filesDic.has_key('tex'):
            if not os.path.exists(texturePath):
                cmd = "install -d -m 755 {0}".format(texturePath)
                cmdList.append(cmd)

            for texFile in filesDic['tex']:
                cmd = "cp -rf {0} {1}".format(texFile, texturePath)
                cmdList.append(cmd)

        if checkBoxList[hasKeyList['images']].checkState() == QtCore.Qt.Checked and filesDic.has_key('images'):
            imgPath = texturePath.replace('/tex/v01', '/v01')
            if not os.path.exists(imgPath):
                cmd = "install -d -m 755 {0}".format(imgPath)
                cmdList.append(cmd)

            for imgFile in filesDic['images']:
                cmd = "cp -rf {0} {1}".format(imgFile, imgPath)
                cmdList.append(cmd)

        if checkBoxList[hasKeyList['proxy']].checkState() == QtCore.Qt.Checked and filesDic.has_key('proxy'):
            imgPath = texturePath.replace('/tex/v01', '/proxy/v01')
            if not os.path.exists(imgPath):
                cmd = "install -d -m 755 {0}".format(imgPath)
                cmdList.append(cmd)

            for imgFile in filesDic['proxy']:
                cmd = "cp -rf {0} {1}".format(imgFile, imgPath)
                cmdList.append(cmd)

        if checkBoxList[hasKeyList['mari']].checkState() == QtCore.Qt.Checked and filesDic.has_key('mari'):
            mariPath = texturePath.replace('/tex/v01', 'mari')

            if not os.path.exists(mariPath):
                cmd = "install -d -m 755 {0}".format(mariPath)
                cmdList.append(cmd)

            cmd = "cp -rf {0} {1}".format(filesDic['mari'], mariPath)
            cmdList.append(cmd)

    shaderPath = os.path.join('/show', showComboBox.currentText(), "asset", "shaders", assetNameLineEdit.text(), "txv01/rfm")

    if checkBoxList[hasKeyList['shader_json']].checkState() == QtCore.Qt.Checked or checkBoxList[hasKeyList['shader_xml']].checkState() == QtCore.Qt.Checked:
        if not os.path.exists(shaderPath):
            cmd = "install -d -m 755 {0}".format(shaderPath)
            cmdList.append(cmd)

        if checkBoxList[hasKeyList['shader_xml']].checkState() == QtCore.Qt.Checked and filesDic.has_key('shader_xml'):
            print '%s/%s_txv01.xml' % (shaderPath, assetNameLineEdit.text())
            cmd = "cp -rf {0} {1}".format(filesDic['shader_xml'], '%s/%s_txv01.xml' % (shaderPath, assetNameLineEdit.text()))
            cmdList.append(cmd)

        if checkBoxList[hasKeyList['shader_json']].checkState() == QtCore.Qt.Checked and filesDic.has_key('shader_json'):
            # cmd = "cp -rf {0} {1}".format(filesDic['shader_json'], shaderPath)
            print '%s/%s_txv01.ma' % (shaderPath, assetNameLineEdit.text())
            exportPath = '%s/%s_txv01.ma' % (shaderPath, assetNameLineEdit.text())

            cmd = '/usr/autodesk/maya2017/bin/mayapy %s/Asset/exportShaderBatch.py %s %s' % (scriptRoot, filesDic['shader_json'], exportPath)

            cmdList.append(cmd)

    if checkBoxList[hasKeyList['hair']].checkState() == QtCore.Qt.Checked and filesDic.has_key('hair'):
        hairPath = os.path.join(modelPath.split('/model')[0], 'hair', 'dev', 'scenes')
        if not os.path.exists(hairPath):
            cmd = "install -d -m 755 {0}".format(hairPath)
            cmdList.append(cmd)

        cmd = "cp -rf {0} {1}".format(filesDic['hair'], hairPath)
        cmdList.append(cmd)

    if checkBoxList[hasKeyList['zbrush']].checkState() == QtCore.Qt.Checked and filesDic.has_key('zbrush'):
        zbrushPath = modelPath.replace('/scenes', '/ztil')
        if not os.path.exists(zbrushPath):
            cmd = "install -d -m 755 {0}".format(zbrushPath)
            cmdList.append(cmd)

        cmd = "cp -rf {0} {1}".format(filesDic['zbrush'], zbrushPath)
        cmdList.append(cmd)

    progressDialog = QtWidgets.QProgressDialog(parent = parent,
                                               labelText = "",
                                               minimum = 0,
                                               maximum = len(cmdList))

    progressDialog.forceShow()

    for index, cmd in enumerate(cmdList):
        print cmd, isinstance(cmd, (list, ))
        progressDialog.setValue(index + 1)

        if isinstance(cmd, (list, )):
            progressDialog.setLabelText("Set " + cmd[2])
        else:
            progressDialog.setLabelText("Set " + cmd.split(' ')[-1])

        QtWidgets.QApplication.processEvents()

        process = subprocess.Popen(cmd, env=_env, shell=True)
        process.wait()
            # os.system(cmd)

        # time.sleep(2)

    progressDialog.close()
    
def importToMaya(filesDic):
    print "AssetDataProcess-importToMaya Call"
    try:
        import sgUI
        import rfmDataTemplate
        import rfmShading
    except Exception as e:
        print "try after opening Maya"
        messageBox = QtWidgets.QMessageBox()
        messageBox.setWindowTitle('Warning')
        messageBox.setIcon(QtWidgets.QMessageBox.Information)
        messageBox.setFont(QtGui.QFont("Cantarell", 13))
        messageBox.setText("try after opening Maya\n%s" % e.message)
        messageBox.addButton("OK", QtWidgets.QMessageBox.AcceptRole)

        messageBox.exec_()
        return

    if filesDic.has_key('model'):
        ciClass = sgUI.ComponentImport(Files=[filesDic['model']], World=0)
        ciClass.m_mode = 1
        ciClass.m_display = 3
        ciClass.m_fitTime = True
        ciClass.doIt()

    if filesDic.has_key('shader_json'):
        rfmDataTemplate.importFile(filesDic['shader_json'])

    if filesDic.has_key('shader_xml'):
        rfmShading.importRlf(filesDic['shader_xml'], 'rlfAdd')

currentpath = os.path.abspath(__file__)

def SetupEnvironment(mayaversion):
    _env = os.environ.copy()
    _env['CURRENT_LOCATION'] = os.path.dirname(currentpath)
    _env['MAYA_VER'] = str(mayaversion)
    _env['RMAN_VER'] = '21.4'
    _env['BACKSTAGE_PATH'] = '/netapp/backstage/pub'
    _env['BACKSTAGE_MAYA_PATH'] = '%s/apps/maya2' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_RMAN_PATH'] = '%s/apps/renderman2' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_ZELOS_PATH'] = '%s/lib/zelos' % _env['BACKSTAGE_PATH']

    _env['RMANTREE'] = '%s/applications/linux/RenderManProServer-%s' % (_env['BACKSTAGE_RMAN_PATH'], _env['RMAN_VER'])
    _env['RMS_SCRIPT_PATHS'] = '%s/rfm-extensions/%s' % (_env['BACKSTAGE_RMAN_PATH'], _env['RMAN_VER'])

    _env['MAYA_LOCATION'] = '/usr/autodesk/maya%s' % mayaversion

    if currentpath.find('/WORK_DATA') > -1:
        _env['BACKSTAGE_MAYA_PATH'] = '/WORK_DATA/script_work/maya2'

    module_path = '%s/modules:%s/modules/%s' % (_env['BACKSTAGE_MAYA_PATH'], _env['BACKSTAGE_RMAN_PATH'], _env['RMAN_VER'])
    _env['MAYA_MODULE_PATH'] = module_path

    _env['LD_LIBRARY_PATH'] = '%s:%s/lib/extern/lib' % (_env['LD_LIBRARY_PATH'], _env['BACKSTAGE_PATH'])
    _env['LD_LIBRARY_PATH'] = '%s:%s/lib' % (_env['LD_LIBRARY_PATH'], _env['BACKSTAGE_ZELOS_PATH'])
    _env['LD_LIBRARY_PATH'] = '%s:%s/maya/%s' % (_env['LD_LIBRARY_PATH'], _env['BACKSTAGE_ZELOS_PATH'], _env['MAYA_VER'])
    _env['LD_LIBRARY_PATH'] = '%s:%s/lib' % (_env['LD_LIBRARY_PATH'], _env['RMANTREE'])

    _env['PATH'] = '%s:%s/lib/extern/bin' % (_env['PATH'], _env['BACKSTAGE_PATH'])

    return _env

