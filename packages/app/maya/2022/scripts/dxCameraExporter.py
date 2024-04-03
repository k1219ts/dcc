# -*- coding: utf-8 -*-
import os, glob
import getpass, datetime
import requests
import pprint
import re, json

import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMayaUI as omu

from shiboken2 import wrapInstance
from PySide2 import QtCore, QtGui, QtWidgets

import sgCommon

import DXUSD.Utils as utl
import DXUSD.Message as msg

import DXUSD_MAYA.Exporters as exp
import DXUSD_MAYA.Camera as Cam
import DXUSD_MAYA.Rig as Rig
import DXUSD_MAYA.MUtils as mutl

import dxConfig
from dxname import tag_parser

import pymongo
from bson.dbref import DBRef
from bson import ObjectId
from pymongo import MongoClient

import ui_DxCameraExporter
import ui_DxCameraExporter_layout

if msg.DEV:
    reload(sgCommon)
    reload(Cam)

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


def get_maya_window():
    main_window_ptr = omu.MQtUtil.mainWindow()
    return wrapInstance(long(main_window_ptr), QtWidgets.QWidget)


def getShotPath(show,seq=None,shot=None):
    if show == 'testshot':  show = 'test_shot'
    shotPath = '/show/%s/_2d/shot' % show
    if seq:
        shotPath = os.path.join(shotPath, seq)
        if shot:
            shotPath = os.path.join(shotPath, shot)
    return shotPath


def getLatestPubVersion(show, seq, shot, data_type,plateType=None):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[show]
    if plateType:
        recentDoc = coll.find_one({'show': show,
                                   'shot': shot,
                                   'data_type': data_type,
                                   'task_publish.plateType':plateType},
                                  sort=[('version', pymongo.DESCENDING)])
    else:
        recentDoc = coll.find_one({'show': show,
                                   'shot': shot,
                                   'data_type': data_type},
                                  sort=[('version', pymongo.DESCENDING)])

    if recentDoc:
        return recentDoc['version']
    else:
        return 0


class CameraWizard(QtWidgets.QWizard):
    def __init__(self, parent=get_maya_window()):
        super(CameraWizard, self).__init__(parent)

        self.projectInfo = self.queryProjects('Active', 'in_progres')
        self.projectInfo.append({'name': u'cdh1', 'title': u'외계인new (cdh1)'})

        if msg.DEV:
            self.projectInfo.append({'name': u'pipe', 'title': u'신규파이프라인 (pipe)'})

        self.intro = IntroPage(self)
        self.camPage = CameraPage(self)
        self.layPage = LayoutPage(self)

        self.addPage(self.intro)
        self.setPage(10, self.camPage)
        self.setPage(20, self.layPage)
        #self.addPage(self.options)

    def accept(self):
        if self.field('camera'):
            camOption = self.camPage.getOptionInfo()
            if not(camOption['task']):
                QtWidgets.QMessageBox.information(self, "NO TEAM",
                                                  "FILL TEAM NAME",
                                                  QtWidgets.QMessageBox.Ok)
                return

            frameRange = [int(camOption['startFrame']),
                          int(camOption['endFrame'])]
            cam = mmvExport(camOption, fr=frameRange, type='camera')
            cam.DoIt()
            dbRecord = cam.getDbRecord()

        elif self.field('layout'):
            camOption = self.layPage.getOptionInfo()
            if not(camOption['task']):
                QtWidgets.QMessageBox.information(self, "NO TEAM",
                                                  "FILL TEAM NAME",
                                                  QtWidgets.QMessageBox.Ok)
                return

            frameRange = [int(camOption['startFrame']),
                          int(camOption['endFrame'])]
            cam = mmvExport(camOption, fr=frameRange, type='layout')
            cam.DoIt()
            dbRecord = cam.getDbRecord()

            dbRecord['task_publish']['plateType'] = 'layout'
            dbRecord['sub_camera_id'] = []

            # # CHECK IF xform HAS KEY
            # # IF KEY: THEN EXPORT
            # # IF NOT: PASS
            # keyAttr = ['translateX', 'translateY', 'translateZ',
            #            'rotateX', 'rotateY', 'rotateZ',
            #            'scaleX', 'scaleY', 'scaleZ']
            # if cmds.keyframe(camOption['camera_list'][0].split('|')[1],
            #                  q=True, at=keyAttr):
            #     dxcFile = 'camera_dxc_main.dxc'
            #     dxcPath = os.path.join(cam.arg.dstdir, dxcFile)
            #
            #     print("main camera dxc node name :", [camOption['camera_list'][0].split('|')[1]])
            #     sgCommon.export_worldAlembic([camOption['camera_list'][0].split('|')[1]],
            #                                  None,
            #                                  int(float(camOption['startFrame'])),
            #                                  int(float(camOption['endFrame'])),
            #                                  1, dxcPath)
            #
            #     abcPath = cmds.getAttr('%s.fileName' % camOption['camera_list'][0].split('|')[1])
            #     dbRecord['files']['camera_path'] = [abcPath]
            #     dbRecord['files']['dxc_path'] = [dxcPath]

            # SUB CAMERA EXPORT
            for index, id in enumerate(camOption['sub_camera_id']):
                if id.startswith('export_'):
                    # MANUALLY CREATE BY ARTIST.
                    # NEED TO EXPORT
                    abcFile = os.path.join(cam.arg.dstdir, 'camera_sub_%s.abc' % id)

                    options = '-v -j "-ef -worldSpace -fr %s %s -s %s' % (camOption['startFrame'],
                                                                          camOption['endFrame'],
                                                                          1)
                    options += ' -rt %s' % id.lstrip('export_')
                    options += ' -f %s"' % abcFile
                    if not(os.path.exists(os.path.dirname(abcFile))):
                        os.makedirs(os.path.dirname(abcFile))

                    mel.eval('AbcExport %s' % options)
                    dbRecord['sub_camera_id'].append({'abc_path':abcFile})

                else:
                    dxcFile = 'camera_dxc_%s.dxc' % id
                    dxcPath = os.path.join(cam.arg.dstdir, dxcFile)

                    sgCommon.export_worldAlembic([camOption['sub_cameras'][index]], None,
                                                 int(float(camOption['startFrame'])),
                                                 int(float(camOption['endFrame'])),
                                                 1, dxcPath)

                    ref = DBRef(collection=camOption['show'],id=ObjectId(id), _extra={'dxc_path':dxcPath})
                    dbRecord['sub_camera_id'].append(ref)

            dbRecord['sub_cameras'] = camOption['sub_cameras']

        # show-isolate mode on
        # for panName in cmds.getPanel(all=True):
        #     if 'modelPanel' in panName:
        #         cmds.isolateSelect(panName, state=1)

        # RIG EXPORT
        if not dbRecord['task_publish']['camera_only']:
            sceneFile = dbRecord['files']['maya_dev_file'][0]
            for rnode in camOption['rig_assets']:
                rig = rigExport(sceneFile, rnode, fr=frameRange)
                keydump = rig.DoIt()

                msg.debug('> RIG DUMP KEY FILE\t:', keydump)
                rigDumpKeys(rnode, keydump).doIt()
                dbRecord['files']['camera_asset_key_path'].append(keydump)

        # show-isolate mode off
        # for panName in cmds.getPanel(all=True):
        #     if 'modelPanel' in panName:
        #         cmds.isolateSelect(panName, state=0)

        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[dbRecord['show']]

        condition = {'data_type': dbRecord['data_type'],
                     'shot': dbRecord['shot'],
                     'version': dbRecord['version'],
                     'task_publish.plateType': dbRecord['task_publish']['plateType']}

        find = coll.find_one(condition)
        if find:    coll.update_one(condition, {'$set': dbRecord})
        else:       coll.insert_one(dbRecord)

        # PUBLISH NOTICE WITH VERSION INFORMATION
        QtWidgets.QMessageBox.information(self, "Camera Publish",
                                          pprint.pformat(dbRecord['files']),
                                          QtWidgets.QMessageBox.Ok)
        print("publish result!!!!")
        print(pprint.pformat(dbRecord['files']))
        print("")

        # CLOSE WIDGET
        super(CameraWizard, self).accept()

    def queryProjects(self, active, status):
        param = {}
        param['api_key'] = API_KEY
        param['category'] = active
        param['status'] = status

        return requests.get("http://10.0.0.51/dexter/search/project.php",
                            params=param).json()

    def getProjectInfo(self):
        return self.projectInfo


class mmvExport:
    def __init__(self, camOption, abcExport=True, overwrite=False,
                 fr=[0, 0], step=1.0, version=None, type='camera'):

        sceneFile = cmds.file(q=True, sn=True)
        self.arg = exp.ACameraExporter()

        self.arg.scene = sceneFile
        self.arg.dxnodes = []
        self.arg.nodes = []
        self.arg.maincam = []
        for cam in camOption['camera_list']:
            self.arg.maincam.append(cmds.ls(cam, sn=True)[0])
        self.arg.frameRange = mutl.GetFrameRange()
        self.arg.overwrite = overwrite
        self.arg.abcExport = abcExport
        self.arg.isStereo = camOption['stereo']
        self.arg.isOverscan = camOption['overscan']
        self.arg.step = step

        self.camOption = camOption
        self.type = type

        # override
        if version: self.arg.ver = version
        if fr != [0, 0]:
            self.arg.frameRange = fr
            self.arg.autofr = False

    def DoIt(self):
        # nodes = []
        if not self.arg.dxnodes:
            self.arg.dxnodes = cmds.ls(type='dxCamera')
        for dxcam in self.arg.dxnodes:
            cShapes = cmds.listRelatives(dxcam, type="camera", f=True, ad=True)
            for shape in cShapes:
                self.arg.nodes.append(cmds.listRelatives(shape, p=True)[0])
        if not self.arg.nodes:
            msg.error('have to select group node.')
        else:
            Cam.CameraExport(self.arg)
            self.arg.overwrite = True
            Cam.CameraCompositor(self.arg)

    def getDbRecord(self):
        #################### DB RECORD & NAMING MUDULE ####################
        version = getLatestPubVersion(show=self.camOption['show'],
                                      seq=self.camOption['sequence'],
                                      shot=self.camOption['shot'],
                                      data_type=self.type,
                                      plateType=self.camOption['plate']) + 1

        dbRecord = {'show': self.arg.show,
                    'sequence': self.arg.seq,
                    'shot': '%s_%s' % (self.arg.seq, self.arg.shot),
                    'task': 'matchmove',
                    'version': version,
                    'data_type': 'camera',
                    'time': datetime.datetime.now().isoformat(),
                    'artist': self.camOption['user'],
                    'files': {'camera_path': [],
                              'imageplane_path': [],
                              'camera_geo_path': [],
                              'camera_loc_path': [],
                              'camera_asset_geo_path': [],
                              'camera_asset_loc_path': [],
                              'camera_asset_key_abc_path': [],
                              'camera_asset_key_path': []
                             },
                     'enabled':True,
                     'sub_cameras': [],
                     'sub_camera_id': [],
                     'task_publish': {'camera_type': 'camera',
                                      'plateType': self.arg.desc,
                                      'stereo': self.arg.isStereo,
                                      'render_width': self.camOption['render_width'],
                                      'render_height': self.camOption['render_height'],
                                      'dx_camera': [],
                                      'camera_only': self.camOption['camera_only'],
                                      'startFrame': self.arg.frameRange[0],
                                      'endFrame': self.arg.frameRange[1],
                                      'overscan': False
                                      }
                     }

        dbRecord['files']['maya_dev_file'] = [self.arg.scene]
        dbRecord['files']['maya_pub_file'] = [self.arg.pubScene]

        overscanValue = cmds.fileInfo("overscan_value", query=True)
        if overscanValue and overscanValue != '1.0':
            dbRecord['task_publish']['overscan'] = True
            dbRecord['task_publish']['overscan_value'] = overscanValue[0]

        mainDxnode = []
        for cam in self.arg.maincam:
            fullPath = cmds.ls(cam, l=True)
            dbRecord['task_publish']['dx_camera'].append(fullPath)
            dxnode = list(filter(None, fullPath[0].split('|')))[0]

        if self.type == 'camera':
            for geom in self.arg.geomfiles:
                geom = geom.replace('.usd', '.abc')
                if not geom in dbRecord['files']['camera_path']:
                    dbRecord['files']['camera_path'].append(geom)

        elif self.type == 'layout':
            for cam in self.camOption['camera_list']:
                node = cam.split('|')[-1]
                for geom in self.arg.geomfiles:
                    if node in geom:
                        geom = geom.replace('.usd', '.abc')
                        if not geom in dbRecord['files']['camera_path']:
                            dbRecord['files']['camera_path'].append(geom)

        if not dbRecord['task_publish']['camera_only']:
            for imp in self.arg.imgPlanefiles:
                imp = imp.replace('.usd', '.abc')
                dbRecord['files']['imageplane_path'].append(imp)

            if self.type == 'layout':
                tmp = self.camOption['camera_list'][0].split('|')
                for dxNode in self.arg.dummyfiles.keys():
                    if not tmp[1] in dxNode:
                        del self.arg.dummyfiles[dxNode]
                        del self.arg.dummyAbc[dxNode]

            for dxNode in self.arg.dummyfiles.keys():
                dummys = self.arg.dummyfiles[dxNode] + self.arg.dummyAbc[dxNode]
                for dummy in dummys:
                    dummy = dummy.replace('.usd', '.abc')

                    kinds = ''
                    if not 'dxCamera' in dummy:
                        kinds = 'asset_'

                    if '.geom' in dummy:
                        dbRecord['files']['camera_%sgeo_path' % (kinds)].append(dummy)
                    else:
                        dbRecord['files']['camera_%sloc_path' % (kinds)].append(dummy)

            dbRecord['files']['imageplane_json_path'] = [self.arg.impAttrfile]
        dbRecord['argv'] = self.arg

        return dbRecord


class rigExport:
    def __init__(self, sceneFile, node, fr=[0, 0], version=None):

        self.arg = exp.ARigShotExporter()
        self.arg.scene= sceneFile
        self.arg.node = node
        self.arg.frameRange = mutl.GetFrameRange()
        self.arg.autofr = True
        self.arg.overwrite = False

        # override
        if version: self.arg.nsver = version
        if fr != [0, 0]:
            self.arg.frameRange = fr
            self.arg.autofr = False

    def DoIt(self):
        Rig.RigShotGeomExport(self.arg)
        self.arg.overwrite = True
        exporter = Rig.RigShotCompositor(self.arg)

        return exporter.arg.master.replace('.usd', '.json')


# Rig Dump keys
class rigDumpKeys:
    def __init__(self, node, filename=None, namespace=True, assetpath=True):
        self.node = node                # dxRig list
        self.filename = filename        # dumpKey json file
        self.namespace = namespace
        self.assetpath = assetpath

    def getAssetpath(self, node):
        filename = cmds.referenceQuery(node, f=True)
        return filename

    def getAttrs(self, node):
        attrs = dict()
        ns_name, node_name = self.getNameSpace(node)
        attrs[node_name] = dict()
        cons = cmds.getAttr(node + '.controllers')
        for con in cons:
            con_name = self.addNamespace(ns_name, con)
            attrs[node_name][con] = cmds.listAttr(con_name, k=True)

            # find null group of controler ( dexter rig only )
            null_name = con.replace("_CON", "_NUL")
            null_node = self.addNamespace(ns_name, null_name)
            if cmds.objExists(null_node):
                attrs[node_name][null_name] = cmds.listAttr(null_node, k=True)
        return attrs

    def addNamespace(self, ns, node):
        if ns:  name = ':'.join([ns, node])
        else:   name = node
        return name

    def getNameSpace(self, nodeName):
        ns_name = ""
        src = nodeName.split(':')
        if len(src) > 1:
            ns_name = ':'.join(src[:-1])
        node_name = src[-1]
        return ns_name, node_name

    def coreKeyDump(self, node, attr):
        connections = cmds.listConnections('%s.%s' % (node, attr), type='animCurve', s=True, d=False)
        msg.debug('> RIG DUMP KEY\t:', node, attr)
        if connections:
            result = dict()
            result['frame'] = cmds.keyframe(node, at=attr, q=True)
            result['value'] = cmds.keyframe(node, at=attr, q=True, vc=True)
            result['angle'] = cmds.keyTangent(node, at=attr, q=True, ia=True, oa=True)
            if cmds.keyTangent(node, at=attr, q=True, wt=True)[0]:
                result['weight'] = cmds.keyTangent(node, at=attr, q=True, iw=True, ow=True)
            result['infinity'] = cmds.setInfinity(node, at=attr, q=True, pri=True, poi=True)
            return result
        else:
            gv = cmds.getAttr('%s.%s' % (node, attr))
            gt = cmds.getAttr('%s.%s' % (node, attr), type=True)
            return {'value':gv, 'type':gt}

    def attributesKeyDump(self, node, attrs):	# attrs=list()
        result = dict()
        for ln in attrs:
            result[ln] = self.coreKeyDump(node, ln)
        return result

    def doIt(self):
        attrdata = self.getAttrs(self.node)

        ns_name, node_name = self.getNameSpace(self.node)
        assetFileName = self.getAssetpath(node=self.node)
        condata = attrdata[node_name]

        for m_con in condata:
            con = self.addNamespace(ns_name, m_con)
            try:
                attrdata[node_name][m_con] = self.attributesKeyDump(node=con, attrs=condata[m_con])
            except:
                msg.debug('> RIG DUMP KEY ERROR\t:', con, condata[m_con])
        if self.namespace:
            attrdata[node_name].update({'_namespace': ns_name})
        if self.assetpath:
            attrdata[node_name].update({'_assetpath': assetFileName})
        attrdata.iterkeys()

        with open(self.filename, 'w') as f:
            json.dump(attrdata, f, indent=4)
            f.close()


class IntroPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(IntroPage, self).__init__(parent)
        self.resize(400,250)
        self.bigFont = QtGui.QFont()
        self.bigFont.setPointSize(14)

        self.label = QtWidgets.QLabel(self)
        self.label.setFont(self.bigFont)

        #self.bigFont.setBold()

        self.label.setText(u'카메라 타입을 선택하세요.')

        self.cameraCheck = QtWidgets.QRadioButton()
        self.cameraCheck.setFont(self.bigFont)
        self.layoutCheck = QtWidgets.QRadioButton()
        self.layoutCheck.setFont(self.bigFont)

        self.cameraCheck.setText('Shot(plate) Camera')
        self.layoutCheck.setText('Layout Camera')

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.cameraCheck)
        self.verticalLayout.addWidget(self.layoutCheck)

        self.register()

    def register(self):
        self.registerField('camera', self.cameraCheck)
        self.registerField('layout', self.layoutCheck)

    def validatePage(self):
        if (not(self.field('camera')) and not(self.field('layout'))):
            QtWidgets.QMessageBox.information(self, u"에러",
                                              u"둘중 하나는 선택하세요.")
            return False
        else:
            print("validate from intro")
            return True

    def nextId(self):
        if self.field('camera'):
            return 10
        elif self.field('layout'):
            return 20
        else:
            return 0


class CameraPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(CameraPage, self).__init__(parent)
        self.ui = ui_DxCameraExporter.Ui_Dialog()
        self.ui.setupUi(self)

        self.titleDic = {}
        self.titleDic['mmv'] = {'name': 'mmv'}
        self.showDic = {}

        for prj in sorted(parent.getProjectInfo(),
                          key=lambda k:k['title']):

            self.ui.showCombo.addItem(prj['title'])
            self.titleDic[prj['title']] = prj
            self.showDic[prj['name']] = prj

        #self.ui.showCombo.insertItem(0, 'mmv')
        # self.ui.showCombo.setCurrentIndex(0)
        self.connectSetting()
        self.prepareInfo()
        self.setPlate()

        # KEYED RANGE CHECK
        # self.keyRangeCheck()


    def connectSetting(self):
        self.ui.showCombo.currentIndexChanged.connect(self.setSeq)
        self.ui.seqCombo.currentIndexChanged.connect(self.setShot)
        self.ui.shotCombo.currentIndexChanged.connect(self.setPlateList)
        self.ui.stereoCheck.toggled.connect(self.set_stereo)
        self.ui.cameraCombo.activated.connect(self.setPlate)

    def keyRangeCheck(self):
        startFrame = float(self.ui.frameRangeFrom.text())
        endFrame = float(self.ui.frameRangeTo.text())
        errorList = []

        for cam in cmds.ls(type='dxCamera'):
            objs = cmds.ls(cam, type='camera', dag=True, ni=True)
            connections = cmds.listConnections(objs, type='animCurve')

            # IF ANY CONNECTIONS THEN MANUAL CHECK TRANSLATE AND ROTATE
            connections = [cmds.listConnections(i, p=1) for i in connections]
            header = connections[0][0].split('.')[0]
            xform = cmds.listRelatives(header, p=True)[0]
            connections.append([xform + '.translate'])
            connections.append([xform + '.rotate'])
            # connections.append([header + '.translate'])
            # connections.append([header + '.rotate'])

            for ln in connections:
                src = ln[0].split('.')
                cmds.setAttr(ln[0], lock=False)
                try:
                    keyStartFrame = cmds.keyframe(ln[0], q=1)[0]
                    keyEndFrame = cmds.keyframe(ln[0], q=1)[-1]

                    cmds.setAttr(ln[0], lock=True)

                    if not((startFrame == keyStartFrame+1) and (endFrame == keyEndFrame-1)):
                        print("errr", i)
                        errorList.append(i)
                except:
                    pass

        if errorList:
            QtWidgets.QMessageBox.information(self, "Key offset error",
                                              "Key offset error\n" + ''.join(errorList),
                                              QtWidgets.QMessageBox.Ok)


    def prepareInfo(self):
        # SHOT INFO
        user = getpass.getuser()
        start_frame = str(cmds.playbackOptions(q=True, min=True))
        end_frame = str(cmds.playbackOptions(q=True, max=True))
        render_width = str(cmds.getAttr("defaultResolution.width"))
        render_height = str(cmds.getAttr("defaultResolution.height"))

        self.ui.userLineEdit.setText(user)
        self.ui.frameRangeFrom.setText(start_frame)
        self.ui.frameRangeTo.setText(end_frame)
        self.ui.resWidthLineEdit.setText(render_width)
        self.ui.resHeightLineEdit.setText(render_height)

        if len(cmds.fileInfo("overscan", query=True)) != 0:
            if cmds.fileInfo("overscan", query=True)[0] == "true":
                self.ui.overscanCheck.setChecked(True)

        if len(cmds.fileInfo("stereo", query=True)) != 0:
            if cmds.fileInfo("stereo", query=True)[0] == "true":
                self.ui.stereoCheck.setChecked(True)
                self.set_stereo(True)

        currentFile = cmds.file(q=True, sn=True)
        currentFile = currentFile.replace('/netapp/dexter/show', '/show')
        currentFile = currentFile.replace('/mach/show/', '/show')

        self.ui.cameraCombo.addItems(self.getCameraList())

        # TODO: NEED TO GET TEAM NAME NOT BASED ON FOLDER STRUCTURE
        try:
            # team = currentFile.split('/')[6]
            # self.ui.teamLineEdit.setText(team)
            self.ui.teamLineEdit.setText('matchmove')

            print(currentFile.split('/'))

            # TODO: GET SHOW / SEQUENCE / SHOT FROM currentFile
            show = currentFile.split('/')[2]
            seq = currentFile.split('/')[6]
            shot = currentFile.split('/')[7]

            if show == 'test_shot':     show = 'testshot'
            showIndex = self.ui.showCombo.findText(self.showDic[show]['title'])
            if showIndex == 0:
                self.setSeq(0)
            self.ui.showCombo.setCurrentIndex(showIndex)
            seqIndex = self.ui.seqCombo.findText(seq)
            self.ui.seqCombo.setCurrentIndex(seqIndex)
            shotIndex = self.ui.shotCombo.findText(shot)
            self.ui.shotCombo.setCurrentIndex(shotIndex)

        except:
            print("set prj seq shot try error")
            self.ui.teamLineEdit.setText('')

    def setSeq(self, index):
        self.ui.seqCombo.clear()
        print(unicode(self.ui.showCombo.currentText()))
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        shotDir = getShotPath(show=project)
        print(prjData, project, shotDir)
        seqs = sorted([i for i in os.listdir(shotDir) if (not (i.startswith('.')))])

        self.ui.seqCombo.clear()
        self.ui.seqCombo.addItems(seqs)

    def setShot(self, index):
        self.ui.shotCombo.clear()
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        seq = unicode(self.ui.seqCombo.currentText())
        seqDir = getShotPath(show=project, seq=seq)

        shots = sorted([i for i in os.listdir(seqDir) if (not (i.startswith('.')))])

        self.ui.shotCombo.clear()
        self.ui.shotCombo.addItems(shots)

    def setPlateList(self):
        self.ui.plateCombo.clear()

        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        seq = unicode(self.ui.seqCombo.currentText())
        shot = self.ui.shotCombo.currentText()
        if not(shot):
            return
        print(project, seq, shot)
        shotDir = getShotPath(show=project, seq=seq, shot=shot)
        # shotDir = shotDir.replace('_3d', '_2d')
        platePath = os.path.join(shotDir, 'plates')
        if os.path.exists(platePath):
            plates = seqs = sorted([i for i in os.listdir(platePath) if (not (i.startswith('.')))])
            if plates:
                self.ui.plateCombo.addItems(plates)

    def set_stereo(self, isStereo):
        if isStereo:
            self.ui.cameraLabel.setText('Left Camera')
            self.ui.rightCamLabel.setText('Right Camera')
            self.ui.rightCamLabel.setEnabled(True)
            self.ui.rightCameraCombo.setEnabled(True)

            self.ui.rightCameraCombo.addItems(self.getCameraList())

        else:
            self.ui.cameraLabel.setText('Camera')
            self.ui.rightCamLabel.setText('Only Stereo')
            self.ui.rightCameraCombo.clear()
            self.ui.rightCamLabel.setEnabled(False)
            self.ui.rightCameraCombo.setEnabled(False)

    def setPlate(self):
        try:
            shotname = self.ui.shotCombo.currentText()
            camname = self.ui.cameraCombo.currentText().split('|')[-1]
            t = re.match('%s_(\S+)_matchmove'%shotname, camname)
            if t:
                platename = t.group(1)
                index = self.ui.plateCombo.findText(platename, QtCore.Qt.MatchExactly)
                if index:
                    self.ui.plateCombo.setCurrentIndex(index)
        except:
            print('no matching plate')

    def getCameraList(self):
        # CAMERA LIST UNDER DXCAMERA
        cTransList = []
        for i in cmds.ls(type='dxCamera'):
            for c in cmds.listRelatives(i, type="camera", fullPath=1, ad=1):
                for cShape in cmds.ls(c, l=True):
                    cTrans = cmds.listRelatives(cShape, p=True, f=True)[0]
                    cTransList.append(cTrans)
        return cTransList

    def getOptionInfo(self):
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        cameraList = []
        cameraList.append(self.ui.cameraCombo.currentText())
        if self.ui.stereoCheck.isChecked():
            cameraList.append(self.ui.rightCameraCombo.currentText())

        rigAssets = []
        if cmds.ls(type='dxRig'):
            for assetKey in cmds.ls(type='dxRig'):
                rigAssets.append(assetKey)
            print('rigAssets',rigAssets)

        optionDic = {}
        optionDic['camera_type'] = 'camera'
        optionDic['show'] = project
        optionDic['sequence'] = seq = unicode(self.ui.seqCombo.currentText())
        optionDic['shot'] = unicode(self.ui.shotCombo.currentText())
        if optionDic['sequence'] == "" or optionDic['shot'] == "":
            return
        optionDic['plate'] = unicode(self.ui.plateCombo.currentText())
        optionDic['task'] = unicode(self.ui.teamLineEdit.text())

        optionDic['startFrame'] = float(self.ui.frameRangeFrom.text())
        optionDic['endFrame'] = float(self.ui.frameRangeTo.text())
        optionDic['render_width'] = self.ui.resWidthLineEdit.text()
        optionDic['render_height'] = self.ui.resHeightLineEdit.text()

        optionDic['overscan'] = self.ui.overscanCheck.isChecked()
        if self.ui.overscanCheck.isChecked():
            try:
                optionDic['overscan_value'] = float(cmds.fileInfo("overscan_value", query=True)[0])
            except:
                optionDic['overscan_value'] = 1.08

        else:
            optionDic['overscan_value'] = 1.0

        optionDic['camera_only'] = self.ui.onlyCamCheck.isChecked()
        optionDic['stereo'] = self.ui.stereoCheck.isChecked()
        optionDic['camera_list'] = cameraList
        optionDic['rig_assets'] = rigAssets


        optionDic['user'] = self.ui.userLineEdit.text()

        return optionDic

    def nextId(self):
        return -1


class LayoutPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(LayoutPage, self).__init__(parent)
        self.ui = ui_DxCameraExporter_layout.Ui_Dialog()
        self.ui.setupUi(self)

        self.titleDic = {}
        self.titleDic['mmv'] = {'name': 'mmv'}
        self.showDic = {}
        for prj in sorted(parent.getProjectInfo(),
                          key=lambda k:k['title']):

            self.ui.showCombo.addItem(prj['title'])
            self.titleDic[prj['title']] = prj
            self.showDic[prj['name']] = prj

        #self.ui.showCombo.insertItem(0, 'mmv')

        self.connectSetting()
        self.prepareInfo()
        #self.ui.showCombo.setCurrentIndex(0)

    def connectSetting(self):
        self.ui.showCombo.currentIndexChanged.connect(self.setSeq)
        self.ui.seqCombo.currentIndexChanged.connect(self.setShot)
        self.ui.stereoCheck.toggled.connect(self.set_stereo)
        self.ui.dxcameraListWidget.currentItemChanged.connect(self.refreshCamera)

    def refreshCamera(self, item, pitem):
        self.ui.cameraCombo.clear()
        dxcamera = [item.text()]
        self.ui.cameraCombo.addItems(self.getCameraList(dxcamera))

        # LET SUB CMAERA WINDOW CAN'T SELECT MAIN CAMERA ITEM
        cRow = self.ui.dxcameraListWidget.row(item)
        if pitem:
            pRow =  self.ui.dxcameraListWidget.row(pitem)
            prevNoMainCam = self.ui.dxcameraListWidget_2.item(pRow)
            prevNoMainCam.setFlags(item.flags())
        noMainCam = self.ui.dxcameraListWidget_2.item(cRow)
        noMainCam.setFlags(QtCore.Qt.NoItemFlags)
        self.ui.dxcameraListWidget_2.clearSelection()

    def getCameraList(self, dxCameras):
        # CAMERA LIST UNDER DXCAMERA
        cTransList = []
        for i in dxCameras:
            for c in cmds.listRelatives(i, type="camera", fullPath=1, ad=1):
                for cShape in cmds.ls(c, l=True):
                    cTrans = cmds.listRelatives(cShape, p=True, f=True)[0]
                    cTransList.append(cTrans)
        return cTransList

    def setSeq(self, index):
        self.ui.seqCombo.clear()
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        shotDir = getShotPath(show=project)
        seqs = sorted([i for i in os.listdir(shotDir) if (not (i.startswith('.')))])

        self.ui.seqCombo.clear()
        self.ui.seqCombo.addItems(seqs)

    def setShot(self, index):
        self.ui.shotCombo.clear()
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        seq = unicode(self.ui.seqCombo.currentText())
        seqDir = getShotPath(show=project, seq=seq)

        shots = sorted([i for i in os.listdir(seqDir) if (not (i.startswith('.')))])

        self.ui.shotCombo.clear()
        self.ui.shotCombo.addItems(shots)

    def prepareInfo(self):
        # SHOT INFO
        user = getpass.getuser()
        start_frame = str(cmds.playbackOptions(q=True, min=True))
        end_frame = str(cmds.playbackOptions(q=True, max=True))
        render_width = str(cmds.getAttr("defaultResolution.width"))
        render_height = str(cmds.getAttr("defaultResolution.height"))

        self.ui.userLineEdit.setText(user)
        self.ui.frameRangeFrom.setText(start_frame)
        self.ui.frameRangeTo.setText(end_frame)
        self.ui.resWidthLineEdit.setText(render_width)
        self.ui.resHeightLineEdit.setText(render_height)

        # DXCAMERA LIST
        for dc in cmds.ls(type='dxCamera'):
            mainItem = QtWidgets.QListWidgetItem(self.ui.dxcameraListWidget)
            mainItem.setText(dc)
            subItem = QtWidgets.QListWidgetItem(self.ui.dxcameraListWidget_2)
            subItem.setText(dc)

        self.ui.dxcameraListWidget_2.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


        # STEREO SETTING CHANGE
        # if len(cmds.fileInfo("stereo", query=True)) != 0:
        #     if cmds.fileInfo("stereo", query=True)[0] == "true":
        #         self.ui.stereoCheck.setChecked(True)
        #         self.set_stereo(True)

        currentFile = cmds.file(q=True, sn=True)
        if currentFile.startswith('/netapp/dexter/show'):
            currentFile = currentFile.replace('/netapp/dexter/show', '/show')
        try:
            team = currentFile.split('/')[6]
            # self.ui.teamLineEdit.setText(team)
            self.ui.teamLineEdit.setText('matchmove')
        except:
            self.ui.teamLineEdit.setText('')

        # TODO: NEED TO GET TEAM NAME NOT BASED ON FOLDER STRUCTURE
        try:
            team = currentFile.split('/')[6]
            # self.ui.teamLineEdit.setText(team)
            self.ui.teamLineEdit.setText('matchmove')

            # TODO: GET SHOW / SEQUENCE / SHOT FROM currentFile
            show = currentFile.split('/')[2]
            seq = currentFile.split('/')[6]
            shot = currentFile.split('/')[7]

            showIndex = self.ui.showCombo.findText(self.showDic[show]['title'])
            if showIndex == 0:
                self.setSeq(0)
            self.ui.showCombo.setCurrentIndex(showIndex)
            seqIndex = seqIndex = self.ui.seqCombo.findText(seq)
            self.ui.seqCombo.setCurrentIndex(seqIndex)
            shotIndex = self.ui.shotCombo.findText(shot)
            self.ui.shotCombo.setCurrentIndex(shotIndex)

        except:
            self.ui.teamLineEdit.setText('')


    def set_stereo(self, isStereo):
        for i in range(self.ui.dxcameraListWidget.count()):
            print(self.ui.dxcameraListWidget.item(i).text())


        if isStereo:
            self.ui.cameraLabel.setText('Left Camera')
            self.ui.rightCamLabel.setText('Right Camera')
            self.ui.rightCamLabel.setEnabled(True)
            self.ui.rightCameraCombo.setEnabled(True)

            dxcamera = [self.ui.dxcameraListWidget.currentItem().text()]
            self.ui.rightCameraCombo.addItems(self.getCameraList(dxcamera))


        else:
            self.ui.cameraLabel.setText('Camera')
            self.ui.rightCamLabel.setText('Only Stereo')
            self.ui.rightCamLabel.setEnabled(False)
            self.ui.rightCameraCombo.setEnabled(False)

    def getOptionInfo(self):
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        # LAYOUT CASE
        # EXPORT ALL CAMERA UNDER dxCamera NODE
        cameraList = self.getCameraList([self.ui.dxcameraListWidget.currentItem().text()])
        # BACK CODE : EXPORT ONLY SELECTED CAMERA
        # cameraList = []
        # cameraList.append(self.ui.cameraCombo.currentText())
        # if self.ui.stereoCheck.isChecked():
        #     cameraList.append(self.ui.rightCameraCombo.currentText())


        # GET MAIN CAMERA'S OBJECT ID IF EXISTS
        print("selected CAMERA : ", self.ui.dxcameraListWidget.currentItem().text())
        mainDxCamera = cmds.ls(self.ui.dxcameraListWidget.currentItem().text())[0]
        print("main camera : ", mainDxCamera)

        # dxRig NAME TO EXPORT
        rigAssets = []

        if cmds.ls(type='dxRig'):
            # NEED TO CHECK IF dxRig IS FROM MAIN or SUB
            if cmds.attributeQuery('objectId', node=mainDxCamera, exists=True):
                print(" have id!!!")
                mainID = cmds.getAttr('%s.objectId' % mainDxCamera)
            else:
                print(" have no id!!!")
                mainID = None
            print("mainID : ", mainID)

            for assetKey in cmds.ls(type='dxRig'):
                print("key", assetKey)
                if cmds.attributeQuery('objectId', node=assetKey, exists=True):
                    rigID = cmds.getAttr('%s.objectId' % assetKey)
                    if rigID == mainID:
                        rigAssets.append(assetKey)

                else:
                    # dxRig node has no id -> means dxRig is part of main camera
                    rigAssets.append(assetKey)
            print('rigAssets',rigAssets)

        subCameras = [i.text() for i in self.ui.dxcameraListWidget_2.selectedItems()]
        subCameraId = []
        for i in subCameras:
            if cmds.attributeQuery('objectId', node=i, exists=True):
                id = cmds.getAttr('%s.objectId' % i)
                subCameraId.append(id)
            else:
                # NO CAMERA ID <- MANUALLY CREATED BY ARTIST.
                # NEED TO EXPORT
                subCameraId.append('export_'+i)



        optionDic = {}
        optionDic['camera_type'] = 'layout'
        optionDic['show'] = project
        optionDic['sequence'] = seq = unicode(self.ui.seqCombo.currentText())
        optionDic['shot'] = unicode(self.ui.shotCombo.currentText())
        if optionDic['sequence'] == "" or optionDic['shot'] == "":
            return
        optionDic['plate'] = u'layout'
        optionDic['task'] = unicode(self.ui.teamLineEdit.text())

        optionDic['startFrame'] = float(self.ui.frameRangeFrom.text())
        optionDic['endFrame'] = float(self.ui.frameRangeTo.text())
        optionDic['render_width'] = self.ui.resWidthLineEdit.text()
        optionDic['render_height'] = self.ui.resHeightLineEdit.text()

        optionDic['overscan'] = self.ui.overscanCheck.isChecked()
        if self.ui.overscanCheck.isChecked():
            try:
                optionDic['overscan_value'] = float(cmds.fileInfo("overscan_value", query=True)[0])
            except:
                optionDic['overscan_value'] = 1.08
        else:
            optionDic['overscan_value'] = 1.0
        optionDic['camera_only'] = self.ui.onlyCamCheck.isChecked()
        optionDic['stereo'] = self.ui.stereoCheck.isChecked()

        optionDic['camera_list'] = cameraList
        optionDic['sub_cameras'] = subCameras
        optionDic['sub_camera_id'] = subCameraId

        optionDic['rig_assets'] = rigAssets

        optionDic['user'] = self.ui.userLineEdit.text()

        return optionDic

    def nextId(self):
        return -1


def showUI():
    mainWidget = CameraWizard()
    mainWidget.show()
