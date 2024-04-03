# -*- coding: utf-8 -*-

import sys
import requests
import dxConfig
import os, platform
from operator import itemgetter
# import pprint

from PySide2 import QtWidgets, QtCore, QtGui

import commands as cmds
from rv import rvtypes, commands, qtutils

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


def chkPlatform():
    # print 'platform:', platform.system()
    return platform.system()


def getSeqList(show_code):
    params = {'api_key': API_KEY,
              'project_code': show_code}

    infos = requests.get("http://%s/dexter/search/sequence.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    if 'ALL' in infos:
        infos.remove('ALL')

    return infos


def getShowList():
    showList = {}
    params = {'api_key': API_KEY,
              'status' : 'in_progres'}
    infos = requests.get("http://%s/dexter/search/project.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

    skipProject = ['test', 'testshot']
    for i in infos:
        if i['code'] in skipProject:
            continue
        showList[i['name']] = i

    return showList


def getTaskList(show_code, shot_name):
    params = {'api_key': API_KEY,
             'project_code': show_code,
             'shot_code': shot_name}

    infos = requests.get("http://%s/dexter/search/snapshot_file.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()
    # pprint.pprint(infos)

    edit, color = getBreakdown(show_code, shot_name)

    tasks = []
    if edit:
        tasks.append('edit')
    for i in infos:
        if not i['process'] in tasks:
            if '.mov' in i['path']:
                tasks.append(i['process'])

    return tasks


def getShot(show_code, sequence_code):
    params = {'api_key': API_KEY,
              'project_code': show_code,
              'sequence_code': sequence_code}
    infos = requests.get("http://%s/dexter/search/shot.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

    shots = []
    for i in infos:
        if not 'Omit' == i['status']:
            shots.append(i)
    return shots


def getMultiShot(showCode, shotName='', depth='', process='', seqCode='', getOrder=False):
    # SEQ검색: showCode, process, seqCode(복수 가능 'ABC|DEF')
    # shot 앞뒤컷 검색: showCode, shotName, depth, subprocess
    # seqCode와 shotName, depth는 동시 검색 불가능

    if 'ALL' == process:   process = ''
    params = {'api_key': API_KEY,
              'project_code': showCode,
              'code': shotName,
              'depth': depth,
              'sequence_code': seqCode,
              'process': process}
    data = requests.get("http://%s/dexter/search/multishot.php" % (dxConfig.getConf('TACTIC_IP')),
                         params=params).json()

    files = []
    if data:
        for i in data['sorted']:
            context = data[i].keys()[0]
            path = data[i][context][0]['path']
            if not '.mov' in path:
                snapshot, color = getSnapshot(showCode, i, '')
                if snapshot:
                    files.append(snapshot[0])
                else:
                    snapshot, colors = getBreakdown(showCode, i)
                    files.append(snapshot[0])
            else:
                files.append(resolvePath(os.path.join('/tactic/assets', path)))

    if getOrder:
        return files, data['sorted']
    else:
        return files


def getMultiShotEditOrder(showCode, editOrder, process=''):
    params = {'api_key': API_KEY,
              'project_code': showCode,
              'edit_order': editOrder,
              'process': process}

    data = requests.get("http://%s/dexter/search/multishot_editorder.php" % dxConfig.getConf('TACTIC_IP'),
                        params=params).json()

    files = []
    if data:
        for i in data['sorted']:
            context = data[i].keys()[0]
            path = data[i][context][0]['path']
            if not '.mov' in path:
                snapshot, color = getSnapshot(showCode, i, '')
                if snapshot:
                    files.append(snapshot[0])
                else:
                    snapshot, colors = getBreakdown(showCode, i)
                    files.append(snapshot[0])
            else:
                files.append(resolvePath(os.path.join('/tactic/assets', path)))

    return files


def getBreakdown(show_code, shot_name, getAll=False):
    params = {'api_key': API_KEY,
              'project_code': show_code,
              'shot_code': shot_name,
              'context': 'publish/edit'}
    infos = requests.get("http://%s/dexter/search/breakdown_file.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

    if not getAll:
        filePath = []
        colors = []
        for i in infos:
            if '.mov' in i['path']:
                filePath.append(resolvePath(i['path']))
        return filePath, colors
    else:
        return infos


def getSnapshot(show_code, shot_name, process=None):
    params = {'api_key': API_KEY,
              'project_code': show_code,
              'shot_code': shot_name}

    if process or process == '':
        params['process'] = process

    infos = requests.get("http://%s/dexter/search/snapshot_file.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()

    if process or process == '':
        filePath = []
        colors = []
        maxCount = 10
        for index, i in enumerate(infos):
            if index >= maxCount:
                break

            if '.mov' in i['path']:
                filePath.append(resolvePath(i['path']))
                if i['task_status'] == 'In-Progress':
                    colors.append([0.75, 1.0, 0.15])
                elif i['task_status'] == 'OK':
                    colors.append([0.0, 1.0, 1.0])
                elif i['task_status'] == 'Retake':
                    colors.append([1.0, 0.0, 0.0])
                elif i['task_status'] == 'Review':
                    colors.append([1.0, 0.8, 0.0])
                elif i['task_status'] == 'Approved':
                    colors.append([0.2, 0.2, 1.0])
                elif i['task_status'] == 'Ready':
                    colors.append([1.0, 1.0, 0.6])
                else:
                    colors.append([1.0, 1.0, 1.0])
        return filePath, colors
    else:
        return infos


def getTaskShot(show_code, shot_name, context):
    params = {'api_key': API_KEY,
              'project_code': show_code,
              'code': shot_name}

    infos = requests.get("http://%s/dexter/search/task_shot.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

    data = {}
    for i in infos:
        if context == i['context']:
            data = i

    return data

def getReviewSupervisor(showCode, startDate, endDate):
    params = {'api_key'      : API_KEY,
              'project_code' : showCode,
              'start_date'   : str(startDate.toString('yyyy-MM-dd')),
              'end_date'     : str(endDate.toString('yyyy-MM-dd'))}

    infos = requests.get("http://%s/dexter/search/review_supervisor.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    infos = sorted(infos, key=itemgetter('pubdate', 'title'), reverse=True)
    return infos

def getReviewSnapshot(showCode, reviewCode):
    params = {'api_key'      : API_KEY,
              'project_code' : showCode,
              'review_code'  : reviewCode}

    infos = requests.get("http://%s/dexter/search/submission_review_supervisor.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    return infos

def getMilestoneName(project):
    params = {'api_key': API_KEY,
              'project_code': project}
    infos = requests.get("http://%s/dexter/search/milestone.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()

    # TODO: NEED SOME FILTER // TOO MANY RESULT
    infos = sorted(infos, key=itemgetter('due_date'), reverse=True)
    infos = sorted(infos, key=itemgetter('name'))

    return infos

def getMilestoneSnapshot(showCode, mileCode):
    params = {'api_key': API_KEY,
              'project_code': showCode,
              'milestone_code': mileCode}

    infos = requests.get("http://%s/dexter/search/milestone_snapshot.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    return infos

def getMilestoneSnapshotAll(showCode, mileCode):
    params = {'api_key': API_KEY,
              'project_code': showCode,
              'milestone_code': mileCode,
              'detail': '1'}

    infos = requests.get("http://%s/dexter/search/milestone_snapshot_all.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    return infos

def getFeedbackTopicName(project):
    params = {'api_key': API_KEY,
              'project_code': project}

    infos = requests.get("http://%s/dexter/search/topic.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    return infos

def getFeedbackTopic(showCode, feedCode):
    params = {'api_key': API_KEY,
              'project_code': showCode,
              'topic_code': feedCode
              }

    infos = requests.get("http://%s/dexter/search/submission_topic.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()
    return infos

def getSourceMediaInfo():
    currentFrame = commands.frame()
    node = commands.sourcesAtFrame(currentFrame)
    info = commands.sourceMediaInfo(node[0])

    return node, info


def clearAnnotate(all=False):
    currentFrame = commands.frame()
    node = commands.sourcesAtFrame(currentFrame)
    currentNode = node[0].replace('_source','')

    nodes = commands.nodes()
    # pprint.pprint(nodes)

    rvPaintNode = []
    for i in nodes:
        if currentNode in i:
            if 'RVPaint' == commands.nodeType(i):
                prop = '%s.paint.nextId' % i
                if commands.propertyExists(prop):
                    value = commands.getIntProperty(prop)
                    print prop, 'value:', value
                    if 0 < value[0]:
                        rvPaintNode.append(i)
    print 'rvPaintNode:', rvPaintNode

    for pNode in rvPaintNode:
        if not all:
            clearPaint(pNode, currentFrame)
        else:
            for prop in commands.properties(pNode):
                # print prop
                if 'frame' in prop and not 'redo' in prop:
                    frame = prop.split(':')[1].replace('.order','')
                    print 'frame:', frame
                    clearPaint(pNode, frame)


def frameOrderName (node, frame):
    return '%s.frame:%s.order' % (node, frame)


def frameOrderRedoStackName (node, frame):
    return '%s.frame:%s.redo' % (node, frame)


def clearPaint(node, frame):
    upropName = frameOrderName(node, frame)
    rpropName = frameOrderRedoStackName(node, frame)

    print 'upropName:', upropName
    print 'rpropName:', rpropName

    if commands.propertyExists(upropName):
        if not commands.propertyExists(rpropName):
            commands.newProperty(rpropName, commands.StringType, 1)

        u = commands.getStringProperty(upropName)
        r = commands.getStringProperty(rpropName)

        print 'u before:', u
        print 'r before:', r

        for i in range(len(u)-1, -1, -1):
            r.append(u[i])
        u = []

        print 'u after:', u
        print 'r after:', r

        commands.setStringProperty(upropName, u, True)
        commands.setStringProperty(rpropName, r, True)


def resolvePath(path):
    if 'Darwin' == chkPlatform():
        if not os.path.isfile(path):
            path = '/opt/' + path
    # print 'resolvePath:', path
    return path


class ReviewConfig(QtCore.QObject):
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.projectDic = {}
        self.revPrjDic = {}

        infos = getShowList()
        for key in infos.keys():
            self.projectDic[key] = infos[key]
            self.revPrjDic[infos[key]['code']] = infos[key]

        self.colorScheme = {'Waiting': QtGui.QColor('#D7D7D7'), 'Omit': QtGui.QColor('#707070'),
                            'Hold': QtGui.QColor('#9D9D9D'), 'Ready': QtGui.QColor('#F7F6CE'),
                            'In-Progress': QtGui.QColor('#CAE1CA'), 'Retake': QtGui.QColor('#E17F81'),
                            'Re-Scan': QtGui.QColor('#E17F81'), 'Changed': QtGui.QColor('#E17F81'),
                            'Review': QtGui.QColor('#A0E6B0'), 'OK': QtGui.QColor('#83D8DE'),
                            'Approved': QtGui.QColor('#6F8BCA'), '': QtGui.QColor(),
                            'NoVFX': QtGui.QColor('#000000'), None: QtGui.QColor()}

        self.statusOrder = {'Approved': 0, 'OK': 1, 'Review': 2, 'In-Progress': 3,
                            'Ready': 4, 'Retake': 5, 'Waiting': 6, 'Hold': 7,
                            'Omit': 8}

        self.teamColor = {'matchmove': QtGui.QColor('#DA8CDD'), 'model': QtGui.QColor('#978CDD'),
                          'creature': QtGui.QColor('#8CA9DD'), 'animation': QtGui.QColor('#8CCDDD'),
                          'texture': QtGui.QColor('#8CDDC5'), 'lighting': QtGui.QColor('#8CDD9B'),
                          'mattepaint': QtGui.QColor('#BFDD8C'), 'fx': QtGui.QColor('#DDC38C'),
                          'comp': QtGui.QColor('#DD8C8C'), 'rnd': QtGui.QColor('#8B9AA5'),
                          'previz': QtGui.QColor('#B0BBDA'), 'edit': QtGui.QColor('#DCD2C3'),
                          'di': QtGui.QColor('#DCC3C3'), 'ani' : QtGui.QColor('#8CCDDD'),
                          'concept': QtGui.QColor('#000000'), 'publish' : QtGui.QColor('#FFFFFF'),
                          '': QtGui.QColor(), None: QtGui.QColor()}

# check platform
chkPlatform()
