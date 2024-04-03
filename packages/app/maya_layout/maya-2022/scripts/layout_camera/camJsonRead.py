# -*- coding: utf-8 -*-
####################################################
#          coding by RND youkyoung.kim             #
####################################################
import os, sys
import json
from maya import cmds

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

class CameraRead(object):
    def __init__(self):
        self.camAttrJson()

    def camAttrJson(self):
        # current show and layout.json exist check
        readjson = {}
        self.model = ''
        show = mayaScene()
        # show = 'wgf'
        if show:
            jsondir, jsonfile = jsonFileCheck(show)
            if os.path.exists(jsonfile):
                readjson = self.readJsonFile(jsonfile)
                cameralist = cameraList()
                self.cameraAttSet(cameralist, readjson)
                self.model = readjson['cameraModel']
        else:
            messageBox('Current Scene : Not find show !!')

    def readJsonFile(self, filename=None):
        # json file read
        f = open(filename, 'r')
        js = json.loads(f.read())
        f.close()
        return js

    def cameraAttSet(self, cams=None, readjson=None):
        # camera attribute setting
        for cam in cams:
            camshape = cmds.listRelatives(cam, shapes=True)[0]
            # cmds.setAttr(camshape + '.focalLength', readjson['focalLength'])
            cmds.setAttr(camshape+'.verticalFilmAperture', readjson['verticalFilmAperture'])
            cmds.setAttr(camshape+'.horizontalFilmAperture', readjson['horizontalFilmAperture'])
            cmds.setAttr(camshape+'.overscan', readjson['overscan'])
            cmds.setAttr(camshape+".displayGateMaskColor", 0, 0, 0)
            cmds.setAttr(camshape+".displayGateMaskOpacity", 1.0)

    def cameraModel(self):
        # json camera model value get
        cammodel = str(self.model)
        return cammodel

#-------------------------------------------------------------
def mayaScene():
    # current maya scene show get
    mayascene = cmds.file(q=True, sn=True)
    if not mayascene == '':
        if mayascene.find('show') > 0:
            show = (mayascene.split('show')[-1]).split('/')[1]
        elif mayascene.startswith('X:'):
            show = (mayascene.split('X')[-1]).split('/')[1]
        else:
            show = ''
        return show
    else:
        return None

def jsonFileCheck(showname=None):
    # current json dir, file value setting
    #  window, linux file path
    # jsondir = os.path.normpath(jsondir)
    # jsonfile = os.path.normpath(jsonfile)
    jsondir = '/show/%s/prev/camera' %showname
    if sys.platform == 'win32':
        jsondir = jsondir.replace('/show/', 'X:/')
    jsonfile = '%s/layoutcam.json' % jsondir
    return jsondir, jsonfile

def cameraList():
    # current scene camera list
    startcam = [u'front', u'persp', u'side', u'top']
    noncam = cmds.listCameras(p=True)
    cameralist = list(set(noncam) - set(startcam))
    cameralist.sort()
    return cameralist

def messageBox(messages='information message', icons= 'warning', buttons=['OK']):
    # warning messagebox
    titles = 'Warning !!'
    msg = '%s    ' % messages
    bgcolor = [0.9, 0.6, 0.6]
    cmds.confirmDialog(title=titles, message=msg,
                       messageAlign='center', icon=icons,
                       button=buttons, backgroundColor=bgcolor)
