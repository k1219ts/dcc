__author__ = 'gyeongheon.jeong'

import os

from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools

import maya.cmds as cmds
import maya.mel as mm

def CF_optionVars():
    if cmds.optionVar( ex = "CF_scaleMin" ) == 0:
        cmds.optionVar( fv = ("CF_scaleMin",0.5) )
    
    if cmds.optionVar( ex = "CF_scaleMax" ) == 0:
        cmds.optionVar( fv = ("CF_scaleMax",1.0) )
        
    if cmds.optionVar( ex = 'CF_AttachToCurve') == 0:
        cmds.optionVar( iv = ('CF_AttachToCurve', 1) )
    
    if cmds.optionVar( ex = 'CF_ConSeg' ) == 0:   
        cmds.optionVar( iv = ('CF_ConSeg', 5) )
        
    if cmds.optionVar( ex = 'CF_SubConSeg' ) == 0:   
        cmds.optionVar( iv = ('CF_SubConSeg', 4) )
       
    if cmds.optionVar( ex = 'CF_EmmiRate' ) == 0:
        cmds.optionVar( iv = ('CF_EmmiRate', 50) )
     
    if cmds.optionVar( ex = 'CF_RanSpeed' ) == 0:
        cmds.optionVar( iv = ('CF_RanSpeed', 0.5) )
        
    if cmds.optionVar( ex = 'CF_lifespan' ) == 0:
        cmds.optionVar( iv = ('CF_lifespan', 5) )
        
    if cmds.optionVar( ex = 'CF_Gweight' ) == 0:
        cmds.optionVar( iv = ('CF_Gweight', 1) )

usrProfile = mm.eval('getenv("USERPROFILE")')
mayaVersion = "2014-x64"

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
UIFILE = os.path.join(CURRENTPATH, "CF_Instancer_NP.ui")

_win = None


def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


def runUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = CF_Instancer_NP()
    _win.show()
    _win.resize(400,200)

class CF_Instancer_NP(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(CF_Instancer_NP, self).__init__(parent)

        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)
        setup_ui(ui, self)

        self.connectSignals()
        self.InitGUI()

    def InitGUI(self):
        CF_optionVars()
        self.scaleItem = "randScale"
        self.indexItem = "randIndex"

        AtcToCurve = cmds.optionVar( q = 'CF_AttachToCurve')
        Conseg = cmds.optionVar( q = 'CF_ConSeg' )
        SubConSeg = cmds.optionVar( q = 'CF_SubConSeg' )

        self.scaleMin = cmds.optionVar( q = "CF_scaleMin" )
        self.scaleMax = cmds.optionVar( q = "CF_scaleMax" )
        self.ParticleList = cmds.ls( type='nParticle' )
        ATCstate = QtCore.Qt.Checked if AtcToCurve == 1 else QtCore.Qt.Unchecked

        self.CF_Instancer_checkBox.setCheckState(ATCstate)
        self.CF_Instancer_NumConSeg.setValue(Conseg)
        self.CF_Instancer_NumConSubSeg.setValue(SubConSeg)
        self.CF_Instancer_MinVal.setValue(self.scaleMin)
        self.CF_Instancer_MaxVal.setValue(self.scaleMax)
        self.CF_Instancer_comboBox.clear()
        self.CF_Instancer_comboBox.addItems( self.ParticleList )


    def connectSignals(self):
        self.CF_Instancer_actionSet_Default.triggered.connect(self.setDefault)
        self.CF_Instancer_MakeInstanceBtn.clicked.connect(self.CF_MakeInstance)
        self.CF_Instancer_CrtFlowBtn.clicked.connect(self.CF_CreateFlowCurve)
        self.CF_Instancer_checkBox.stateChanged.connect(self.OptionVarChange)
        self.CF_Instancer_NumConSeg.valueChanged.connect(self.OptionVarChange)
        self.CF_Instancer_NumConSubSeg.valueChanged.connect(self.OptionVarChange)
        self.CF_Instancer_MinVal.valueChanged.connect(self.OptionVarChange)
        self.CF_Instancer_MaxVal.valueChanged.connect(self.OptionVarChange)


    def OptionVarChange(self):
        AtcToCurve = int( self.CF_Instancer_checkBox.checkState() )
        cmds.optionVar( iv = ('CF_AttachToCurve', AtcToCurve) )

        Conseg = int( self.CF_Instancer_NumConSeg.value() )
        cmds.optionVar( iv = ('CF_ConSeg', Conseg) )

        SubConSeg = int( self.CF_Instancer_NumConSubSeg.value() )
        cmds.optionVar( iv = ('CF_SubConSeg', SubConSeg) )

        scaleMin = self.CF_Instancer_MinVal.value()
        cmds.optionVar( fv = ("CF_scaleMin", scaleMin) )

        scaleMax = self.CF_Instancer_MaxVal.value()
        cmds.optionVar( fv = ("CF_scaleMax", scaleMax) )

    def setDefault(self):
        cmds.optionVar( fv = ("CF_scaleMin",0.5) )
        cmds.optionVar( fv = ("CF_scaleMax",1.0) )
        cmds.optionVar( iv = ('CF_AttachToCurve', 1) )
        cmds.optionVar( iv = ('CF_ConSeg', 5) )
        cmds.optionVar( iv = ('CF_SubConSeg', 4) )
        cmds.optionVar( iv = ('CF_EmmiRate', 50) )
        cmds.optionVar( iv = ('CF_RanSpeed', 0.5) )
        cmds.optionVar( iv = ('CF_lifespan', 5) )
        cmds.optionVar( iv = ('CF_Gweight', 1) )

        self.InitGUI()

    def CF_CreateFlowCurve(self, *args):
        FlowGrpName = str( self.CF_Instancer_grpName.text() )
        AtcToCurve = cmds.optionVar( q = 'CF_AttachToCurve')
        Conseg = cmds.optionVar( q = 'CF_ConSeg' )
        SubConSeg = cmds.optionVar( q = 'CF_SubConSeg' )
        EmmiRate = cmds.optionVar( q = 'CF_EmmiRate' )
        RanSpeed = cmds.optionVar( q = 'CF_RanSpeed' )
        lifespan = cmds.optionVar( q = 'CF_lifespan' )
        Gweight = cmds.optionVar( q = 'CF_Gweight' )
        mm.eval('source "{0}/flowAlongCurves_nParticle.mel";'.format(CURRENTPATH))
        mm.eval('flowAlongCurves_NP "%s" %d %d %d %f %f %f %f;' % ( FlowGrpName, Conseg,
                                                                    SubConSeg, AtcToCurve,
                                                                    EmmiRate, RanSpeed,
                                                                    lifespan, Gweight))
        self.InitGUI()

    def CF_MakeInstance(self, *args):
        selInsObjs = cmds.ls(sl = True)
        particleShape = str(self.CF_Instancer_comboBox.currentText())
        expr_str = "{0}.goalPP = 0;\n"
        expr_str += "vector $idVector = {0}.particleId;\n"
        expr_str += "vector $randomPosition = dnoise( $idVector * 10.0 ) * 100.0;\n"
        expr_str += "{0}.randomPosition = $randomPosition;\n"
        expr_str += "float $randomMotionSpeed = {0}.randomMotionSpeed;\n"
        expr_str += "float $maxDistance = {0}.maxDistance;\n"
        expr_str += "vector $curveOffset =\n"
        expr_str += "   dnoise( $randomPosition + ( time * $randomMotionSpeed ) ) *\n"
        expr_str += "   $maxDistance;\n"
        expr_str += "vector $rampValues = {0}.rampValues;\n"
        expr_str += "{0}.curveOffset =\n"
        expr_str += "{0}.curveOffset;\n"
        expr_str += "{0}.goalOffset =\n"
        expr_str += "   $rampValues + $curveOffset;\n\n"
        expr_str += "float $sc = rand( {1}, {2} );\n"
        expr_str += "{0}.randScale = << $sc, $sc, $sc >>;\n\n"
        expr_str += "{0}.randIndex = rand({3})"

        expr_str = expr_str.format( particleShape,
                                    cmds.optionVar( q = "CF_scaleMin" ),
                                    cmds.optionVar( q = "CF_scaleMax" ),
                                    len( selInsObjs ) )

        if selInsObjs == []:
            cmds.confirmDialog( title='Confirm', message='Select Object First',
                                button=['Close'], defaultButton='Close')
        else:
            if cmds.attributeQuery("randScale", node = particleShape, exists = True):
                pass
            else:
                cmds.addAttr(particleShape, ln = "randScale0", dt = "doubleArray" )
                cmds.addAttr(particleShape, ln = "randScale", dt = "doubleArray" )
                cmds.setAttr(particleShape + ".randScale", e = True, keyable = True )
                
                cmds.addAttr(particleShape, ln = "randIndex0", dt = "doubleArray" )
                cmds.addAttr(particleShape, ln = "randIndex", dt = "doubleArray" )
                cmds.setAttr(particleShape + ".randIndex", e = True, keyable = True )
                cmds.dynExpression(particleShape, s = expr_str, c = True )
        
            cmds.particleInstancer( particleShape, addObject = True,
                                    object = selInsObjs, cycle = "None",
                                    cycleStep = 1, cycleStepUnits = "Frames",
                                    levelOfDetail = "Geometry", rotationUnits = "Degrees",
                                    rotationOrder = "XYZ", position = "worldPosition",
                                    scale = self.scaleItem, aimDirection = "velocity",
                                    objectIndex = self.indexItem, age = "age" )
