#coding:utf-8
import xlrd2
import xlwt
import os
import getpass
import requests
import datetime
import argparse
import sys

# Tractor
import tractor.api.author as author

from Define import *
import DBConfig

def getShowCode(showName):
    projectName = showName
    requestParam = dict() # eqaul is requestParm = {}
    requestParam['api_key'] = API_KEY
    requestParam['name'] = projectName
    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=TACTIC_IP), params=requestParam)

    projectInfo = responseData.json()[0]
    return projectInfo['code'], projectInfo['sync']

# Tractor
TRACTOR_IP = '10.0.0.25'
PORT = 80
serviceKey = "Editorial"
project = "export"
tier = "cache"
tags = ""

def writeXlsRow(sheet, rowIndex, rowData):
    for col in range(len(rowData)):
        sheet.write(rowIndex, col, rowData[col])

def spool(xlsFileName, isShotNameBurn, isEffectBurn, uploadSeqMov, plateType, editOrder):
    showName = xlsFileName.split('/')[3]

    # DB
    coll = DBConfig.db[showName.lower()]

    # Job Setup
    job = author.Job()
    job.title = '(EDITORIAL) %s' % os.path.basename(xlsFileName)
    job.comment = 'sourcefile : ' + xlsFileName
    job.service = serviceKey
    job.tier = tier
    job.tags = [tags]
    job.projects = [project]
    job.priority = 100

    if not os.path.exists(xlsFileName):
        assert False, "Not found XLS File"

    plateListExcel = xlrd2.open_workbook(xlsFileName)
    sheet = plateListExcel.sheet_by_name('scan_list')

    editModifyListExcel = xlwt.Workbook(encoding='utf-8')
    editModifyExcelSheet = editModifyListExcel.add_sheet('scan_list')

    rootTask = author.Task(title='Shot Cutout')
    job.addChild(rootTask)

    splitXlsFile = xlsFileName.split('/')

    shotMovFileList = []
    writeXlsRow(editModifyExcelSheet, 0, sheet.row_values(0))
    row = 1
    editOrderIndexList = []
    # for row in range(1, sheet.nrows):
    while row < sheet.nrows - 1:
        rowData = sheet.row_values(row)
        issue = rowData[Column2.ISSUE.value]
        clipName = rowData[Column2.CLIP_NAME.value]
        if len(clipName) >= 25:
            clipName = clipName[:25]
        startTC = rowData[Column2.TC_IN.value]
        endTC = rowData[Column2.TC_OUT.value]
        clipPlateType = rowData[Column2.TYPE.value]
        editOrderIndex = rowData[Column2.EDIT_ORDER.value]

        if editOrderIndex in editOrderIndexList:
            row += 1
            continue

        movFile = rowData[Column2.XML_NAME.value].replace('.xml', '.mov')
        if not os.path.exists(movFile):
            movFile = rowData[Column2.XML_NAME.value].replace('.xml', '.mp4')
        movDirectory = os.path.dirname(movFile)
        # movDirectory = '/prod_nas/__DD_PROD/EMD/edit/20210205/15th_source'

        burnInDirName = 'burnin'

        task = author.Task(title='{DSTMOV}'.format(DSTMOV=clipName))
        # First EditMOV to Shot jpg
        command = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.7.1', '--']
        command += ['rvio', '-v', movFile]
        command += ['-t', '%s-%s' % (int(rowData[Column2.MOV_CUT_IN.value]), int(rowData[Column2.MOV_CUT_IN.value]) + int(rowData[Column2.CLIP_DURATION.value]) - 1)]
        command += ['-in709', '-outsrgb']

        dstJpgDir = os.path.join(movDirectory, '_shot_burn_in_', '%d_%s' % (editOrderIndex, clipName))
        dstJpgRule = os.path.join(dstJpgDir, '%s_%s.#.jpg' % (clipName, clipPlateType))

        if not os.path.exists(dstJpgDir):
            os.makedirs(dstJpgDir)
        command += ['-o', '%s' % dstJpgRule]
        task.addCommand(author.Command(argv=command, service=serviceKey))

        # Seconds burn in Using Nuke
        nukeCommand = ['/backstage/dcc/DCC', 'rez-env', 'nuke-12.2.4', '--']
        nukeCommand += ['nukeX', '-i', '-t', '-X', 'Write1']
        nukeCommand += ['/backstage/dcc/packages/ext/otiotoolkit/scripts/burnInUsingNuke.py']
        nukeCommand += ['--jpgdir', '%s' % dstJpgDir]
        nukeCommand += ['--shotNameBurn', 'False']
        nukeCommand += ['--effectBurn', 'False']
        print ' '.join(nukeCommand)
        task.addCommand(author.Command(argv=nukeCommand, service=serviceKey))

        # Third Jpg To Mov
        burnInMovFileName = os.path.join(movDirectory, '_shot_burn_in_', '%d_%s.mov' % (editOrderIndex, clipName))
        movCommand = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
        movCommand += ['ffmpeg_converter', '-i', os.path.join(dstJpgDir, burnInDirName)]
        movCommand += ['-r', '%s' % rowData[Column2.MOV_CUT_FPS.value]]
        movCommand += ['-o', burnInMovFileName]
        movCommand += ['-c', 'h264']
        task.addCommand(author.Command(argv=movCommand, service=serviceKey))
        shotMovFileList.append(burnInMovFileName)

        rootTask.addChild(task)
        editOrderIndexList.append(editOrderIndex)
        row += 1

    author.setEngineClientParam(hostname=TRACTOR_IP, port=80, user=getpass.getuser(), debug=True)
    # print job.as_tcl()
    job.spool()
    author.closeEngineClient()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # mov file name
    argparser.add_argument('-f', '--file', dest='file', type=str, required=True, help='xls file name')
    argparser.add_argument('-sb', '--shotNameBurn', dest='shotNameBurn', type=str, default="True", help='this argument default true, but set argument is false')
    argparser.add_argument('-eb', '--effectBurn', dest='effectBurn', type=str, default="True", help='this argument default true, but set argument is false')
    argparser.add_argument('-vs', '--velozSeqMov', dest='velozSeqMov', type=str, help='this argument default true, but set argument is false')
    argparser.add_argument('-pt', '--plateType', dest='plateType', type=str, help='plate type choice [main, src]')
    argparser.add_argument('-eo', '--editOrder', dest='editOrder', type=str, help='how to changed edit order?')

    args, unknown = argparser.parse_known_args(sys.argv)
    shotNameBurn = True
    if args.shotNameBurn == "False":
        shotNameBurn = False

    effectBurn = True
    if args.effectBurn == "False":
        effectBurn = False

    velozSeqMov = True
    if args.velozSeqMov == "False":
        velozSeqMov = False

    editOrder = True
    if args.editOrder == "False":
        editOrder = False

    spool(args.file, shotNameBurn, effectBurn, velozSeqMov, args.plateType, editOrder)
