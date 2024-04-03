#encoding=utf-8
#!/usr/bin/env python

"""
RenderMan For Maya Tractor Job Spool Tool

LAST RELEASE:
- 2017.08.29 : start
- 2017.09.08 : rebuild ui
- 2017.12.01 : stereo camera control
"""

import Qt
import Qt.QtGui as QtGui
import Qt.QtWidgets as QtWidgets
import Qt.QtCore as QtCore
import os

if Qt.__qt_version__ > "5.0.0":
    import shiboken2 as shiboken
else:
    import shiboken as shiboken

import maya.OpenMayaUI as apiUI
from config import *
from configMaya import *

current = os.path.abspath(__file__)
root = os.path.dirname(current)
uiroot = os.path.join(root, 'qtui')

UITEMP = os.path.join(os.path.expanduser('~'), '.rfmsubmitter.gui')


def messageBox(Title='Warning',
               Message = 'warning message',
               Icon = 'warning', # warning, question, information, critical
               Button = ['OK', 'Cancel'],
               bgColor = [.5,.5,.5]):
    msg = cmds.confirmDialog(title=Title, message='%s    ' % Message,
                             messageAlign='center', icon=Icon,
                             button=Button, backgroundColor=bgColor)
    return msg

def getMayaWindow():
    ptr = apiUI.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

class styleSheets():
    lineEdit_Red = 'QLineEdit {color: rgb(200,50,1)}'
    lineEdit_Normal = 'QLineEdit {color: rgb(200,200,200)}'
    spinBox_Red = 'QSpinBox {color: rgb(200,50,1)}'
    spinBox_Normal = 'QSpinBox {color: rgb(200,200,200)}'


# import UI
from UI4.spooltool_v03 import Ui_JobScriptMainWindow
from UI4.renderManRIS_option_v01 import Ui_Form as RenderManRISUI
# from UI4.stereo_option import Ui_Form as StereoUI
import UI4.stereo_option
reload(UI4.stereo_option)
StereoUI = UI4.stereo_option.Ui_Form


class JobScriptMainWindow(QtWidgets.QMainWindow):
    """
    RenderManForMaya Tractor Job Spool Tool
    """
    def __init__( self, parent = getMayaWindow() ):
        super( JobScriptMainWindow, self ).__init__( parent )

        self.currentRenderer = cmds.getAttr('defaultRenderGlobals.currentRenderer')
        if self.currentRenderer != 'renderManRIS':
            return

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_JobScriptMainWindow()
        self.ui.setupUi(self)


        # default ui setup
        self.setWindowTitle('RfM Submitter')
        self.ui.jobParametersFrame.hide()

        # center of the screen
        mayaWindow = getMayaWindow()
        self.move(mayaWindow.frameGeometry().center() - self.frameGeometry().center())

        self.style = styleSheets()

        # current renderer
        self.ui.currentRenderer_label.setText(self.currentRenderer)
        self.ui.currentRenderer_label.setStyleSheet('QLabel {color: rgb(77,129,200)}')

        # engine
        self.ui.tractorEngine_comboBox.addItems(GetEngineList())

        # user
        self.ui.userLineEdit.setText(getpass.getuser())

        # camera
        self.stereo_widget = ''
        if StereoRenderStatus():
            class StereoOptWidget(QtWidgets.QDialog):
                def __init__(self, parent):
                    super(StereoOptWidget, self).__init__(parent)
                    self.ui = StereoUI()
                    self.ui.setupUi(self)
            self.stereo_widget = StereoOptWidget(self)
            self.ui.StereoLayout.addWidget(self.stereo_widget, 1)
            # bind
            self.stereo_widget.ui.sectionStereo.clicked.connect(self.sectionStereoProc)

        # render options widget
        self.opt_widget = ''
        class RenderOptWidget(QtWidgets.QDialog):
            def __init__(self, parent, currentRenderer):
                super(RenderOptWidget, self).__init__(parent)
                if currentRenderer == "renderManRIS":
                    self.ui = RenderManRISUI()
                self.ui.setupUi( self )
        self.opt_widget = RenderOptWidget(self, self.currentRenderer)
        self.ui.OptionLayout.addWidget(self.opt_widget)

        # set default ui
        self.setDefaultUI()

        # file setup
        self.fileSetup()

        # frame range setup
        self.ui.frameRange_lineEdit.setText(FrameRange())

        # resolution setup
        self.resolutionSetup()

        # command binding
        self.ui.resolutionFull.clicked.connect(self.resolutionSetupProc)
        self.ui.resolutionProxy.clicked.connect(self.resolutionSetupProc)
        self.ui.outputVersion_lineEdit.textChanged.connect(self.outputImagesProc)
        self.ui.browsePreCompFile.clicked.connect(self.setPreCompFileProc)
        self.ui.sendJobButton.clicked.connect(self.renderProc)
        self.ui.recoveryButton.clicked.connect(self.recoveryProc)
        self.ui.closeButton.clicked.connect(self.closeWindow)

        self.restoreUI()
        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.saveCurrentUI()
            self.close()

    # set up ui
    def setDefaultUI( self ):
        # icons
        out_aimConstraint_icon = QtGui.QIcon(os.path.join(root, 'icons', 'out_aimConstraint.png'))
        SP_FileDialogBack_icon = QtGui.QIcon(os.path.join(root, 'icons', 'SP_FileDialogBack.png'))
        SP_FileDialogForward_icon = QtGui.QIcon(os.path.join(root, 'icons', 'SP_FileDialogForward.png'))
        rvRenderGlobals_icon = QtGui.QIcon(os.path.join(root, 'icons', 'rvRenderGlobals.png'))
        SP_DirOpenIcon_icon = QtGui.QIcon(os.path.join(root, 'icons', 'SP_DirOpenIcon.png'))
        SP_TrashIcon_icon = QtGui.QIcon(os.path.join(root, 'icons', 'SP_TrashIcon.png'))

        self.ui.getTacticFrameButton.setIcon(out_aimConstraint_icon)
        self.ui.setFrameFromTactic.setIcon(SP_FileDialogBack_icon)
        self.ui.browsePreCompFile.setIcon(SP_DirOpenIcon_icon)

        # renderMan
        self.opt_widget.ui.ribgenLimit_spinBox.setValue(1)
        self.opt_widget.ui.shutterAngle_spinBox.setValue(180)
        self.opt_widget.ui.motionBlur_checkBox.setChecked(True)
        self.opt_widget.ui.cameraBlur_checkBox.setChecked(True)
        minsamples = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_minsamples')
        self.opt_widget.ui.minSamples_spinBox.setValue(minsamples)
        maxsamples = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_maxsamples')
        self.opt_widget.ui.maxSamples_spinBox.setValue(maxsamples)
        self.opt_widget.ui.checkPoint_spinBox.setValue(20)
        self.opt_widget.ui.incremental_checkBox.setChecked(True)
        if cmds.attributeQuery('rman__torattr___denoise', n='renderManRISGlobals', ex=True):
            denoise = cmds.getAttr('renderManRISGlobals.rman__torattr___denoise')
            self.opt_widget.ui.denoise_comboBox.setCurrentIndex(denoise)

    def sectionStereoProc(self):
        if not self.stereo_widget:
            return
        right = self.stereo_widget.ui.sectionRight.isChecked()
        setRight = False if (right == True) else True;
        self.stereo_widget.ui.sectionRight.setChecked(setRight)
        left = self.stereo_widget.ui.sectionLeft.isChecked()
        setLeft = False if (left == True) else True;
        self.stereo_widget.ui.sectionLeft.setChecked(setLeft)

    # file set up     ----------------------------------------------------------
    def fileSetup(self):
        self.ui.workspace_lineEdit.setReadOnly(True)
        self.ui.sceneFile_lineEdit.setReadOnly(True)

        currentScene = cmds.file(q=True, sn=True)
        mayaProj = MayaProjectPath()

        self.ui.workspace_lineEdit.setText(mayaProj)

        relativeScene = RelativePath(currentScene)
        self.ui.sceneFile_lineEdit.setText(relativeScene)
        if os.path.isabs(relativeScene):
            self.ui.workspace_lineEdit.setStyleSheet(self.style.lineEdit_Red)
        else:
            self.ui.workspace_lineEdit.setStyleSheet(self.style.lineEdit_Normal)

        # output images
        self.ui.outputVersion_lineEdit.setText(LastVersion())
        output = 'images' + '/' + str(self.ui.outputVersion_lineEdit.text())
        abs_output = AbsolutePath(output)
        if os.path.exists(abs_output):
            self.ui.outputVersion_lineEdit.setStyleSheet(self.style.lineEdit_Red)
        else:
            self.ui.outputVersion_lineEdit.setStyleSheet(self.style.lineEdit_Normal)

        # recovery version
        verList = VersionList()
        self.ui.recoveryVersionCbox.addItems(verList)
        if verList:
            self.ui.recoveryVersionCbox.setCurrentIndex(\
                self.ui.recoveryVersionCbox.findText(verList[-1]))

    def outputImagesProc(self):
        abs_output = self.getOutputImagePath()
        if os.path.exists(abs_output):
            self.ui.outputVersion_lineEdit.setStyleSheet(self.style.lineEdit_Red)
        else:
            self.ui.outputVersion_lineEdit.setStyleSheet(self.style.lineEdit_Normal)

    def getOutputImagePath(self, recovery=None):
        if not recovery:
            version = self.ui.outputVersion_lineEdit.text()
        else:
            version = self.ui.recoveryVersionCbox.currentText()
        output = 'images/' + version
        return AbsolutePath(output)

    def setPreCompFileProc(self):
        startPath = cmds.workspace(rd=True, q=True)
        current = str(self.ui.precomp_lineEdit.text())
        if current:
            startPath = os.path.dirname(current)

        fn = cmds.fileDialog2(fileMode = 1,
                              caption = 'Select PreComp Nuke File',
                              okCaption = 'select',
                              fileFilter = 'Nuke (*.nk)',
                              startingDirectory = startPath)
        if fn:
            self.ui.precomp_lineEdit.setText(RelativePath(fn[0]))
        cmds.showWindow('JobScriptMainWindow')

    # resolution setup ---------------------------------------------------------
    def resolutionSetup(self):
        width  = cmds.getAttr('defaultResolution.width')
        height = cmds.getAttr('defaultResolution.height')
        self.ui.width_spinBox.setValue(width)
        self.ui.height_spinBox.setValue(height)

    def resolutionSetupProc(self):
        width  = self.ui.width_spinBox.value()
        height = self.ui.height_spinBox.value()
        # Full
        if self.ui.resolutionFull.isChecked():
            if width < 1800:
                self.ui.width_spinBox.setValue(width * 2)
                self.ui.height_spinBox.setValue(height * 2)
        if self.ui.resolutionProxy.isChecked():
            if width > 1800:
                self.ui.width_spinBox.setValue(width / 2)
                self.ui.height_spinBox.setValue(height / 2)

    def getVersion(self, recovery=None):
        if recovery:
            return self.ui.recoveryVersionCbox.currentText()
        else:
            return self.ui.outputVersion_lineEdit.text()

    def getOptions(self, recovery=None):
        opts = {}
        # file
        opts['m_mayaproj'] = self.ui.workspace_lineEdit.text()
        opts['m_version'] = self.getVersion(recovery)
        opts['m_outdir']   = self.getOutputImagePath(recovery)

        # common
        opts['m_engine']    = str(self.ui.tractorEngine_comboBox.currentText())
        try:
            data = getConfig['TractorEngine']
            opts['m_port'] = int(data[data.index(opts['m_engine'])+1])
            opts['m_cloudJob'] = False
        except:
            data = getConfig['CloudEngine']
            opts['m_port'] = int(data[data.index(opts['m_engine']) + 1])
            opts['m_cloudJob'] = True

        opts['m_renderer']  = self.currentRenderer
        opts['m_priority']  = 100
        opts['m_envkey']    = str(mel.eval('rman getPref DefaultEnvKey'))
        opts['m_maxactive'] = 0
        opts['m_range']     = str(self.ui.frameRange_lineEdit.text())
        opts['m_by']        = int(self.ui.frameRangeBy_spinBox.value())

        # resolution
        opts['m_width']  = self.ui.width_spinBox.value()
        opts['m_height'] = self.ui.height_spinBox.value()

        # user
        opts['m_user'] = self.ui.userLineEdit.text()

        # Stereo Camera
        if self.stereo_widget:
            opts['m_stereo'] = {
                'right': self.stereo_widget.ui.sectionRight.isChecked(),
                'left': self.stereo_widget.ui.sectionLeft.isChecked()
            }

        # options
        opts['m_ribgenLimit'] = self.opt_widget.ui.ribgenLimit_spinBox.value()
        opts['m_ribgenOnly'] = int(self.opt_widget.ui.ribgenOnly_checkBox.isChecked())
        opts['m_shutterAngle'] = self.opt_widget.ui.shutterAngle_spinBox.value()
        opts['m_motionBlur'] = int(self.opt_widget.ui.motionBlur_checkBox.isChecked())
        opts['m_cameraBlur'] = int(self.opt_widget.ui.cameraBlur_checkBox.isChecked())
        opts['m_tracedBlur'] = int(self.opt_widget.ui.tracedBlur_checkBox.isChecked())
        opts['m_minsamples'] = self.opt_widget.ui.minSamples_spinBox.value()
        opts['m_maxsamples'] = self.opt_widget.ui.maxSamples_spinBox.value()
        opts['m_checkPoint'] = self.opt_widget.ui.checkPoint_spinBox.value()
        if recovery:
            opts['m_incremental'] = 1
        else:
            opts['m_incremental'] = int(self.opt_widget.ui.incremental_checkBox.isChecked())
        opts['m_denoise'] = int(self.opt_widget.ui.denoise_comboBox.currentIndex())
        opts['m_denoiseFilter'] = cmds.getAttr('renderManRISGlobals.rman__torattr___denoiseFilter')
        opts['m_denoiseaov'] = int(self.opt_widget.ui.denoise_aov_checkBox.isChecked())
        opts['m_denoiseStrength'] = self.opt_widget.ui.denoise_strength.value()

        # post render
        opts['m_precompFile'] = str(AbsolutePath(str(self.ui.precomp_lineEdit.text())))

        # recovery
        opts['m_recovery'] = int(str(recovery)=='True')

        return opts

    # command binding ----------------------------------------------------------
    def closeWindow(self):
        self.saveCurrentUI()
        self.close()

    def restoreUI(self):
        if not os.path.exists(UITEMP):
            return
        f = open(UITEMP, 'r')
        data = json.loads(f.read())

        self.ui.tractorEngine_comboBox.setCurrentIndex(self.ui.tractorEngine_comboBox.findText(data['m_engine']))

        if data.has_key('m_ribgenLimit'):
            self.opt_widget.ui.ribgenLimit_spinBox.setValue(data['m_ribgenLimit'])
        if data.has_key('m_ribgenOnly'):
            self.opt_widget.ui.ribgenOnly_checkBox.setChecked(data['m_ribgenOnly'])
        if data.has_key('m_shutterAngle'):
            self.opt_widget.ui.shutterAngle_spinBox.setValue(data['m_shutterAngle'])
        if data.has_key('m_motionBlur'):
            self.opt_widget.ui.motionBlur_checkBox.setChecked(data['m_motionBlur'])
        if data.has_key('m_cameraBlur'):
            self.opt_widget.ui.cameraBlur_checkBox.setChecked(data['m_cameraBlur'])
        if data.has_key('m_tracedBlur'):
            self.opt_widget.ui.tracedBlur_checkBox.setChecked(data['m_tracedBlur'])

        minsamples = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_minsamples')
        self.opt_widget.ui.minSamples_spinBox.setValue(minsamples)
        maxsamples = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_maxsamples')
        self.opt_widget.ui.maxSamples_spinBox.setValue(maxsamples)
        if data.has_key('m_checkPoint'):
            self.opt_widget.ui.checkPoint_spinBox.setValue(data['m_checkPoint'])
        if data.has_key('m_incremental'):
            self.opt_widget.ui.incremental_checkBox.setChecked(data['m_incremental'])

    def saveCurrentUI(self):
        f = open(UITEMP, 'w')
        opts = self.getOptions()
        json.dump(opts, f)
        f.close()

    def renderProc(self, recover=None):
        currentScene = cmds.file(q=True, sn=True)
        if not currentScene:
            messageBox('Save File', 'You have to save file', 'critical', ['OK'], [.6,0,0])
            self.closeWindow()
            return

        opts = self.getOptions(recover)

        # Debug
        pprint.pprint(opts)

        exec('import %s_script' % self.currentRenderer)
        jobClass = eval('%s_script.JobMain(opts)' % self.currentRenderer)
        jobClass.doIt()

        self.closeWindow()

    # recovery button command
    def recoveryProc(self):
        # any version exist
        if not VersionList():
            massage = "Shot don't has any recovery version"
            cmds.confirmDialog(m=massage)
            return

        self.renderProc(recover=True)

def show():
    if cmds.window('JobScriptMainWindow', exists=True, q=True):
        cmds.deleteUI('JobScriptMainWindow')

    renderer = cmds.getAttr('defaultRenderGlobals.currentRenderer')
    if renderer != 'renderManRIS':
        return

    if GetFormatList(renderer):
        JobScriptMainWindow()
    else:
        messageBox(Message='Not Support Renderer', Button=['OK'])
