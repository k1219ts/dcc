#encoding=utf-8
#-------------------------------------------------------------------------------
#
#   Dexter CGSupervisor
#
#		daeseok.chae	cds7031@gmail.com
#
#	Dexter Outliner Callback
#
#	2020.08.19
#-------------------------------------------------------------------------------
#
#	RigToUsd
#	    - RigToUsdReference, UsdReferenceToRig
#   Version Controller
#       - dxRigVersionControl
#
#-------------------------------------------------------------------------------

import os
import string
import maya.cmds as cmds
from PySide2 import QtWidgets
from PySide2 import QtGui

def dxRigVersionControl():
    '''
    reference file path of dxRig use, other rig version change.
    also if same file using, change other rig too. ( suggest )
    :return:
    '''
    dxRigNodes = cmds.ls(sl=True, type='dxRig')

    # first, get reference file of selected dxRig
    selectedReferenceFileDict = {}

    for dxRig in dxRigNodes:
        referenceFilePath = cmds.referenceQuery(dxRig, filename = True)
        selectedReferenceFileDict[dxRig] = {"origin":referenceFilePath, "other":"", "item":None}

    # Find Other Version
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("select rig version")
    vLayout = QtWidgets.QVBoxLayout(dialog)

    for nodeName in selectedReferenceFileDict.keys():
        referenceDir = os.path.dirname(selectedReferenceFileDict[nodeName]['origin'])
        excludeVer = os.path.basename(selectedReferenceFileDict[nodeName]['origin'])
        fileList = []
        for rigVerFile in sorted(os.listdir(referenceDir)):
            print rigVerFile
            if ".mb" in rigVerFile and not rigVerFile.startswith(".") and rigVerFile != excludeVer:
                fileList.append(rigVerFile.split(".")[0])

        fileList.sort(reverse=True)

        if not fileList:
            selectedReferenceFileDict.pop(nodeName)
            continue

        hLayout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel()
        label.setText(nodeName.split(":")[0])
        hLayout.addWidget(label)
        comboBox = QtWidgets.QComboBox()
        comboBox.addItems(fileList)
        hLayout.addWidget(comboBox)
        selectedReferenceFileDict[nodeName]['item'] = comboBox
        vLayout.addLayout(hLayout)

    hLayout = QtWidgets.QHBoxLayout()
    okBtn = QtWidgets.QPushButton()
    okBtn.setText("OK")
    okBtn.clicked.connect(dialog.accept)
    hLayout.addWidget(okBtn)

    cancelBtn = QtWidgets.QPushButton()
    cancelBtn.setText("CANCEL")
    cancelBtn.clicked.connect(dialog.close)
    hLayout.addWidget(cancelBtn)
    vLayout.addLayout(hLayout)

    if not selectedReferenceFileDict:
        cmds.warning("don't have other rig versions.")
        return

    dialog.show()
    dialog.close()
    geometry = dialog.geometry()
    yPos = QtGui.QCursor.pos().y()
    if QtGui.QCursor.pos().y() + geometry.height() > 1160:
        yPos -= geometry.height()
    dialog.setGeometry(QtGui.QCursor.pos().x(), yPos, geometry.width(), geometry.height())

    retDlg = dialog.exec_()

    if not retDlg:
        return

    # Change Reference Version
    originFileList = {}
    for nodeName in selectedReferenceFileDict.keys():
        selectRigVer = selectedReferenceFileDict[nodeName]['item'].currentText()
        originDir = os.path.dirname(selectedReferenceFileDict[nodeName]['origin'])
        newRigFilePath = os.path.join(originDir, selectRigVer + ".mb")
        print newRigFilePath

        ns = cmds.referenceQuery(nodeName, rfn = True)

        cmds.file(newRigFilePath, loadReference = ns)
        originFileList[selectedReferenceFileDict[nodeName]['origin']] = newRigFilePath

    # find same reference file in other rigNodes
    appendChangeNodes = {}
    for dxRig in cmds.ls(type = "dxRig"):
        curReferencePath = cmds.referenceQuery(dxRig, filename = True)

        print curReferencePath, "in", originFileList.keys()
        curReferencePath = curReferencePath.split("{")[0]
        if curReferencePath in originFileList.keys():
            appendChangeNodes[dxRig] = originFileList[curReferencePath]

    # if other rignodes to same files, suggest change other Nodes?
    if appendChangeNodes:
        print appendChangeNodes
        dialog = QtWidgets.QDialog()
        vLayout = QtWidgets.QVBoxLayout(dialog)
        dialog.setWindowTitle("change other nodes?")

        for nodeName in appendChangeNodes.keys():
            rigVersion = os.path.splitext(os.path.basename(appendChangeNodes[nodeName]))[0]

            label = QtWidgets.QLabel()
            label.setText("%s => %s" % (nodeName.split(":")[0], rigVersion))
            vLayout.addWidget(label)

        hLayout = QtWidgets.QHBoxLayout()
        okBtn = QtWidgets.QPushButton()
        okBtn.setText("OK")
        okBtn.clicked.connect(dialog.accept)
        hLayout.addWidget(okBtn)

        cancelBtn = QtWidgets.QPushButton()
        cancelBtn.setText("CANCEL")
        cancelBtn.clicked.connect(dialog.close)
        hLayout.addWidget(cancelBtn)
        vLayout.addLayout(hLayout)

        dialog.show()
        dialog.close()
        geometry = dialog.geometry()
        yPos = QtGui.QCursor.pos().y()
        if QtGui.QCursor.pos().y() + geometry.height() > 1160:
            yPos -= geometry.height()
        dialog.setGeometry(QtGui.QCursor.pos().x(), yPos, geometry.width(), geometry.height())

        retDlg = dialog.exec_()

        if not retDlg:
            return

        for nodeName in appendChangeNodes.keys():
            rigFile = appendChangeNodes[nodeName]

            ns = cmds.referenceQuery(nodeName, rfn=True)

            cmds.file(rigFile, loadReference=ns)


def rigRepresentCtrl():
    for node in cmds.ls(sl=True):
        ntype = cmds.nodeType(node)
        if ntype == 'pxrUsdReferenceAssembly':
            print '# Action : "%s" pxrUsdReferenceAssembly -> dxRig' % node
            rootNode = cmds.ls(node, l = True)[0].split("|")[1]
            if cmds.nodeType(rootNode) == "dxBlock": # pxrUsdReferenceAssembly in set
                UsdReferenceToRig(node, procType="set")
            else:
                print "# Action : skip"
                # UsdReferenceToRig(node)
        elif ntype == 'dxRig':
            print "# Action : dxRig -> pxrUsd Reference >> '%s'" % node
            RigToUsdReference(node)
        else:
            proxyShape = cmds.ls(node, dag=True, type='pxrUsdProxyShape')
            if proxyShape:
                print "# Action : pxrUsdProxyShape -> dxRig >> '%s'" % node
                UsdReferenceToRig(proxyShape[0])


class UsdReferenceToRig:
    def __init__(self, node, procType = None):
        ntype = cmds.nodeType(node)
        if ntype == 'pxrUsdProxyShape':
            self.proxyShapeProc(node)
        elif ntype == 'pxrUsdReferenceAssembly':
            if procType == "set":
                self.sceneAssemblyProcFromSet(node)
            else:
                self.sceneAssemblyProc(node)

    def proxyShapeProc(self, node):
        trNode  = cmds.listRelatives(node, p=True, f=True)[0]
        if cmds.attributeQuery('refNode', n=trNode, ex=True):
            rfn = cmds.getAttr('%s.refNode' % trNode)
            cmds.file(loadReference=rfn)
            nodes = cmds.referenceQuery(rfn, nodes=True)
            cmds.select(nodes[0])
        else:
            refFile = cmds.getAttr('%s.filePath' % node)
            splitPath = refFile.split('/')
            if 'asset' in splitPath:
                index = splitPath.index('asset')
                assetName = splitPath[index+1]
                assetPath = string.join(splitPath[:index+2] , '/')
                rigPath   = '{DIR}/rig/scenes'.format(DIR=assetPath)
                rigFile   = self.getRigFile(rigPath)
                if rigFile:
                    ns, node = self.referenceImport(rigFile, assetName)
                    dst = ns + ':place_CON'
                    self.constraintSetup(trNode, dst)
                    self.attributeSetup(trNode, node)
                    cmds.select(node)
        cmds.setAttr('%s.visibility' % trNode, 0)


    def getRigFile(self, rigPath):
        files = list()
        if not os.path.exists(rigPath):
            cmds.error('# Error : not found rig scene directory')

        for fn in sorted(os.listdir(rigPath)):
            if '.mb' in fn and not fn.startswith('.'):
                files.append(fn)
        if files:
            return os.path.join(rigPath, files[-1])

    def referenceImport(self, filename, assetName):
        rfile = cmds.file(filename, r=True, iv=True, op='v=0;', ns=assetName+'1')
        rfn   = cmds.referenceQuery(rfile, rfn=True)
        nodes = cmds.referenceQuery(rfn, nodes=True)
        ns    = cmds.referenceQuery(rfn, ns=True)
        return ns[1:], nodes[0]

    def constraintSetup(self, src, dst):
        '''
        Args:
            src : usd reference node
            dst : dxRig place_CON node
        '''
        cmds.pointConstraint(src, dst)
        cmds.orientConstraint(src, dst)
        # cmds.scaleConstraint(src, dst)

        # cmds.setAttr('%s.globalScaleX' % dst, cmds.getAttr('%s.scaleX' % src))
        # cmds.setAttr('%s.globalScaleY' % dst, cmds.getAttr('%s.scaleY' % src))
        # cmds.setAttr('%s.globalScaleZ' % dst, cmds.getAttr('%s.scaleZ' % src))
        cmds.setAttr('%s.globalScale' % dst, cmds.getAttr('%s.scaleX' % src))

    def attributeSetup(self, src, dst):
        '''
        Args:
            src : usd reference node
            dst : dxRig node
        '''
        # src attributes
        if not cmds.attributeQuery('rigNode', n=src, ex=True):
            cmds.addAttr(src, ln='rigNode', dt='string')
        cmds.setAttr('%s.rigNode' % src, dst, type='string')
        if not cmds.attributeQuery('refNode', n=src, ex=True):
            cmds.addAttr(src, ln='refNode', dt='string')
        cmds.setAttr('%s.refNode' % src, cmds.referenceQuery(dst, rfn=True), type='string')

        # dst attributes
        cmds.setAttr('%s.rigType' % dst, 3)
        cmds.setAttr('%s.action' % dst, 0)
        if not cmds.attributeQuery('targetNode', n=dst, ex=True):
            cmds.addAttr(dst, ln='targetNode', dt='string')
        cmds.setAttr('%s.targetNode' % dst, cmds.ls(src, l=True)[0], type='string')

    def sceneAssemblyProcFromSet(self, node):
        parentNode = cmds.listRelatives(node, p=True, f=True)[0]
        if cmds.attributeQuery('refNode', n=node, ex=True):
            rfn = cmds.getAttr('%s.refNode' % node)
            cmds.file(loadReference=rfn)
            nodes = cmds.referenceQuery(rfn, nodes=True)
            cmds.select(nodes[0])
        else:
            refFile = cmds.getAttr('%s.filePath' % node)
            splitPath = refFile.split('/')
            if 'asset' in splitPath:
                index = splitPath.index('asset')
                assetName = splitPath[index+1]
                assetPath = string.join(splitPath[:index+2] , '/')
                rigPath   = '{DIR}/rig/scenes'.format(DIR=assetPath)
                rigFile   = self.getRigFile(rigPath)
                if rigFile:
                    ns, rigNode = self.referenceImport(rigFile, assetName)
                    dst = ns + ':place_CON'
                    self.constraintSetup(node, dst)
                    self.attributeSetup(node, rigNode)
                    rigNode = cmds.parent(rigNode, parentNode)
                    cmds.select(rigNode)

        cmds.setAttr('%s.visibility' % node, 0)

    def sceneAssemblyProc(self, node):
        if cmds.attributeQuery('refNode', n=node, ex=True):
            rfn = cmds.getAttr('%s.refNode' % node)
            cmds.file(loadReference=rfn)
            nodes = cmds.referenceQuery(rfn, nodes=True)
            cmds.select(nodes[0])
        else:
            refFile = cmds.getAttr('%s.filePath' % node)
            splitPath = refFile.split('/')
            if 'asset' in splitPath:
                index = splitPath.index('asset')
                assetName = splitPath[index+1]
                assetPath = string.join(splitPath[:index+2] , '/')
                rigPath   = '{DIR}/rig/scenes'.format(DIR=assetPath)
                rigFile   = self.getRigFile(rigPath)
                if rigFile:
                    ns, rigNode = self.referenceImport(rigFile, assetName)
                    dst = ns + ':place_CON'
                    self.constraintSetup(node, dst)
                    self.attributeSetup(node, rigNode)
                    cmds.select(rigNode)
        cmds.setAttr('%s.visibility' % node, 0)


class RigToUsdReference:
    def __init__(self, node):
        ntype = cmds.nodeType(node)
        if ntype == 'dxRig':
            self.rigProc(node)

    def rigProc(self, node):
        if not cmds.attributeQuery('targetNode', n=node, ex=True):
            return
        targetNode = cmds.getAttr('%s.targetNode' % node)

        rfn = cmds.referenceQuery(node, rfn=True)
        cmds.file(unloadReference=rfn)

        cmds.setAttr('%s.visibility' % targetNode, 1)
        cmds.select(targetNode)




#-------------------------------------------------------------------------------
def pxrUsdLodVariantSet(name):
    for n in cmds.ls(sl=True, type='pxrUsdReferenceAssembly'):
        if cmds.attributeQuery('usdVariantSet_lodVariant', n=n, ex=True):
            cmds.setAttr('%s.usdVariantSet_lodVariant' % n, name, type='string')
