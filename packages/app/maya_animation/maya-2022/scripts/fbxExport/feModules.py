# encoding:utf-8
import os
import json
import logging
import string
import subprocess
import maya.cmds as cmds
import maya.mel as mel
import aniCommon
import sgCamera
import sgCommon

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FbxExport():
    @staticmethod
    def createPolyImagePlane():
        logger.debug(u'Convert imageplane to polygon')
        new_imageplane = list()
        cameras = cmds.ls(type='camera')
        starttime = cmds.playbackOptions(q=True, min=True)
        endtime = cmds.playbackOptions(q=True, max=True)

        for cam in cameras:
            imageplanes = cmds.listConnections(cam, type='imagePlane', d=False)
            if not imageplanes:
                continue
            new, tmp = sgCamera.createPolyImagePlane(cam, imageplanes)
            for i, polymesh in enumerate(new):
                matrixs, frames = sgCommon.getMtx(tmp[i], starttime, endtime, 1)
                sgCommon.setMtx(polymesh, matrixs, frames)
            new_imageplane += new
            cmds.select(new)
            cmds.filterCurve()
            cmds.delete(tmp)
        return new_imageplane


    @staticmethod
    def getAssemblies():
        logger.debug(u'Get top level nodes')
        topNode = cmds.ls(assemblies=True)
        rmList = list()
        for i in topNode:
            iShape = cmds.listRelatives(i, shapes=True)
            if len(cmds.ls(iShape)) == 1:
                if cmds.nodeType(i) == 'dxRig' or \
                                cmds.nodeType(iShape) == 'camera':
                    rmList.append(i)
        for i in rmList:
            topNode.remove(i)
        return topNode


    def __init__(self):
        self._version = 'FBX201400'
        self._character = str()
        self._scale = float()
        self._output = str()
        self._objects = list()


    @property
    def version(self):
        return self._version


    @property
    def character(self):
        return self._character


    @property
    def scale(self):
        return self._scale


    @property
    def output(self):
        return self._output


    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value


    def getPivot(self):
        """Get center pivot of dxRig character's world controler

        :param character: dxRig character
        :return: A list of pivot values
        """
        namespace = aniCommon.getNameSpace(self.character)
        moveCon = string.join([namespace, "move_CON"], ":")

        pivot = cmds.xform(moveCon, q=True, rp=True, ws=True)
        return pivot


    def getTextures(self):
        """Get file textures

        :return: A list of texture file full names
        """
        textures = list()
        fileList = cmds.ls(type='file')
        for file in fileList:
            textureName = cmds.getAttr(file + '.fileTextureName')
            if textureName not in textures:
                textures.append(textureName)
        return textures


    def getImageplane(self):
        """Get imageplanes and connected camera

        :return: {imageplane : camera}
        """
        imagePlaneDic = dict()
        imps = cmds.ls(type='imagePlane')
        for imp in imps:
            impNode = cmds.imagePlane(imp, q=True, name=True)
            cam = cmds.imagePlane(imp, q=True, camera=True)
            if cam:
                imagePlaneDic[impNode[0]] = cam
        return imagePlaneDic


    def getImagePath(self):
        """Get imageplane filename

        :return: A list of image file paths
        """
        images = list()
        impDic = self.getImageplane()
        for imp in impDic:
            imageName = cmds.getAttr(imp + '.imageName')
            images.append(imageName)
        return images


    def setPath(self):
        """Set .fbx file name and image path

        :return: [image path, fbx filename]
        """
        shotName = self.output.split(os.sep)[-1]
        fbxFile = os.path.join(self.output, shotName + "_env.fbx")
        imageDir = os.path.join(self.output, "image")
        if not os.path.exists(imageDir):
            os.mkdir(imageDir)
        return [imageDir, fbxFile]


    def writejson(self, data, file):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
            f.close()


    def copyImage(self, fileList):
        imageDir = self.setPath()[0]
        for oriImage in fileList:
            oriImageDir = os.sep.join(oriImage.split(os.sep)[:-1])
            logger.debug(u'Copy image files : {}'.format(oriImageDir))
            logger.debug(u'To : {}'.format(imageDir))
            p = subprocess.Popen(['cp', '-r', oriImageDir, imageDir])
            p.wait()


    def addObjects(self):
        new_polyplanes = FbxExport.createPolyImagePlane()
        self.objects += new_polyplanes


    def scaleScene(self):
        """Group all object, and edit scale

        :return: Group node name
        """
        logger.debug(u'Scale scene size : {0}'.format(self.scale))
        pivot = self.getPivot()
        groupNode = cmds.group(self.objects, n='env_GRP')
        cmds.xform(groupNode,
                   rp=(pivot[0], pivot[1], pivot[2]),
                   sp=(pivot[0], pivot[1], pivot[2]),
                   ws=True)
        cmds.scale(self.scale,
                   self.scale,
                   self.scale,
                   groupNode)
        cmds.xform(groupNode,
                   t=(pivot[0]*-1, pivot[1]*-1, pivot[2]*-1),
                   ws=True)
        return groupNode


    def setDefault(self, groupNode):
        cmds.xform(groupNode, t=(0, 0, 0), s=(1, 1, 1))
        childs = cmds.listRelatives(groupNode, fullPath=True)
        cmds.parent(childs, w=True)
        cmds.delete(groupNode)


    def export(self):
        fbxFile = self.setPath()[1]
        images = self.getImagePath()
        textures = self.getTextures()

        logger.debug(u'Export fbx')
        logger.debug(u'Output : {0}'.format(fbxFile))

        self.copyImage(images)
        self.copyImage(textures)

        self.addObjects()
        groupNode = self.scaleScene()
        cmds.select(groupNode, add=False)
        evalString  = 'FBXExportFileVersion "{version}";\n'
        evalString += 'FBXExportInAscii -v true;\n'
        evalString += 'FBXExportConstraints -v true;\n'
        evalString += 'FBXExport -f "{file}" -s;'
        evalString = evalString.format(version=self.version,
                                       file=fbxFile)
        print evalString
        mel.eval(evalString)
        self.setDefault(groupNode)
        return True

