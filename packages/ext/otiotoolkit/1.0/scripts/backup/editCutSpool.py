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

# Tactic
from tactic_client_lib import TacticServerStub
import dxConfig

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

# SLC
# xlsFileName = '/prod_nas/__DD_PROD/SLC/edit/20201111/S30_MET/S30_MET_v1_CGDI_201111_(Resolve).xls' # SLC MET SEQUENCE
# xlsFileName = '/prod_nas/__DD_PROD/SLC/edit/20201116/S39_DAT_locationEdit/Kang_action_1114_toEditor.xls' # SLC DAT SEQUENCE

# CDH1
# xlsFileName = '/prod_nas/__DD_PROD/CDH/edit/during_pre/201111_keyshot/201112_from_edit_opt/1_s041_KEY_IMAGE.xls' # CDH KEY SEQUENCE

# EMD
# xlsFileName = '/prod_nas/__DD_PROD/EMD/new/edit/20201130/201130_preCG/XML/Emergency_A01_PreCG_Guide_CAR_EDL_201130.xls'
# xlsFileName = '/prod_nas/__DD_PROD/EMD/new/edit/20201130/201130_preCG/XML/Emergency_A01_PreCG_Guide_OFF_EDL_201130.xls'
# xlsFileName = '/prod_nas/__DD_PROD/EMD/new/edit/20201130/201130_preCG/XML/Emergency_A01_PreCG_Guide_RAT_EDL_201130.xls'

# PRAT2
# xlsFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201130/haejeok_002_S010_201130.xls'
# xlsFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201130/haejeok_002_S044_201130.xls'
# xlsFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201130/haejeok_002_S083_201130.xls'

# Tractor
TRACTOR_IP = '10.0.0.25'
PORT = 80
serviceKey = "Editorial"
# project = "comp"
# tier = "comp"
# tags = "2d"
project = "export"
tier = "cache"
tags = ""

# Tactic
TACTIC_IP = dxConfig.getConf("TACTIC_IP")
API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
login = 'daeseok.chae'
password = 'dexter#1322'

def writeXlsRow(sheet, rowIndex, rowData):
    for col in range(len(rowData)):
        sheet.write(rowIndex, col, rowData[col])

def spool(xlsFileName, isShotNameBurn, isEffectBurn, uploadSeqMov, plateType, modifyEditOrder):
    showName = xlsFileName.split('/')[3]
    project_code, sync = getShowCode(showName.lower())
    print showName.lower(), project_code

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

    # Tactic Login
    tactic = TacticServerStub(login=login, password=password, server=TACTIC_IP, project=project_code)
    tactic.start()

    splitXlsFile = xlsFileName.split('/')
    editDate = datetime.datetime.now().strftime('%Y/%m/%d')
    if 'prod_nas' in splitXlsFile:
        editIndex = splitXlsFile.index('edit')
        editDate = splitXlsFile[editIndex + 1].split('_')[0]
        print editDate

    shotMovFileDict = {}
    shotNameList = []
    onlyShotNameList = ['DMO_1455']
    rowIndex = 1
    writeXlsRow(editModifyExcelSheet, 0, sheet.row_values(0))
    row = 1
    # for row in range(1, sheet.nrows):
    while row < sheet.nrows:
        rowData = sheet.row_values(row)
        shotName = rowData[Column2_org.SHOT_NAME.value]
        issue = rowData[Column2_org.ISSUE.value]
        clipName = rowData[Column2_org.CLIP_NAME.value]
        startTC = rowData[Column2_org.TC_IN.value]
        endTC = rowData[Column2_org.TC_OUT.value]
        clipPlateType = rowData[Column2_org.TYPE.value]
        isModify = False

        if not shotName:
            row += 1
            continue

        if onlyShotNameList and shotName not in onlyShotNameList:
            row += 1
            continue

        shot_search_type = '%s/shot' % project_code

        if "Omit" in issue:
            # But Check Tactic shot
            queryData = tactic.query(shot_search_type, filters=[('code', shotName)])
            if queryData:
                data = {}
                data['status'] = "Omit"
                print "Veloz Omit Shot :", data
                shotBuildKey = tactic.build_search_key(shot_search_type, shotName)
                print shotBuildKey
                try:
                    tactic.update(shotBuildKey, data)
                except:
                    tactic.abort()
            row += 1
            continue

        movFile = rowData[Column2_org.XML_NAME.value].replace('.xml', '.mov')
        if not os.path.exists(movFile):
            movFile = rowData[Column2_org.XML_NAME.value].replace('.xml', '.mp4')
        movDirectory = os.path.dirname(movFile)

        if shotName in shotNameList:
            row += 1
            continue

        shotDetailDict = {}
        index = 0
        if row + 1 != sheet.nrows:
            while row + 1 < sheet.nrows and sheet.row_values(row + 1)[Column2_org.SHOT_NAME.value] == shotName:
                otherRowData = sheet.row_values(row + 1)
                shotDetailDict[index] = {Column2_org.CLIP_NAME.name.lower(): otherRowData[Column2_org.CLIP_NAME.value],
                                         Column2_org.TYPE.name.lower(): otherRowData[Column2_org.TYPE.value],
                                         Column2_org.ISSUE.name.lower(): otherRowData[Column2_org.ISSUE.value]}
                row += 1

        # edit modify listup
        if "top" in issue or "end" in issue:
            # edit range modify
            print "edit Range Modify", shotName, issue
            writeXlsRow(editModifyExcelSheet, rowIndex, rowData)
            rowIndex += 1
            isModify = True

        findItem = DBConfig.getData(coll, shotName, clipName, startTC, endTC)
        if not findItem and not isModify:
            print "# new shot", shotName
            writeXlsRow(editModifyExcelSheet, rowIndex, rowData)
            rowIndex += 1

        burnInDirName = 'burnin'

        seq = shotName.split('_')[0]
        sequence_search_type = '%s/sequence' % project_code
        sequence_data = {
            'code': seq,
            'name': seq,
            'original_seq_code': seq,
            'roll': 'R1',
            'status': 'Waiting',
            'pipeline_code': sequence_search_type
        }

        task = author.Task(title='{DSTMOV}'.format(DSTMOV=shotName))
        # First EditMOV to Shot jpg
        command = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.7.1', '--']
        command += ['rvio', '-v', movFile]
        command += ['-t', '%s-%s' % (int(rowData[Column2_org.MOV_CUT_IN.value]), int(rowData[Column2_org.MOV_CUT_IN.value] + rowData[Column2_org.MOV_CUT_DURATION.value]) - 1)]
        command += ['-in709', '-outsrgb']

        padding = len(str(int(rowData[Column2_org.MOV_CUT_IN.value])))
        if padding < len(str(int(rowData[Column2_org.MOV_CUT_IN.value]) + int(rowData[Column2_org.MOV_CUT_DURATION.value]) - 1)):
            padding = len(str(int(rowData[Column2_org.MOV_CUT_IN.value]) + int(rowData[Column2_org.MOV_CUT_DURATION.value]) - 1))

        dstJpgDir = os.path.join(movDirectory, '_shot_burn_in_', '%s' % shotName)
        dstJpgRule = os.path.join(dstJpgDir, '%s.' % shotName + '%0' + str(padding) + 'd.jpg')

        if not os.path.exists(dstJpgDir):
            os.makedirs(dstJpgDir)
        command += ['-o', '%s' % dstJpgRule]
        task.addCommand(author.Command(argv=command, service=serviceKey))

        # First.5 MOV to Shot wav
        soundCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.7.1', '--']
        soundCommand += ['rvio', '-v', movFile]
        soundCommand += ['-t', '%s-%s' % (int(rowData[Column2_org.MOV_CUT_IN.value]),
                                     int(rowData[Column2_org.MOV_CUT_IN.value] + rowData[Column2_org.MOV_CUT_DURATION.value]) - 1)]

        wavFilePath = os.path.join(movDirectory, '_shot_burn_in_', '%s.wav' % shotName)
        soundCommand += ['-o', '%s' % wavFilePath]
        task.addCommand(author.Command(argv=soundCommand, service=serviceKey))

        # Seconds burn in Using Nuke
        nukeCommand = ['/backstage/dcc/DCC', 'rez-env', 'nuke-12.2.4', '--']
        nukeCommand += ['nukeX', '-i', '-t', '-X', 'Write1']
        nukeCommand += ['/backstage/dcc/packages/ext/otiotoolkit/scripts/burnInUsingNuke.py']
        nukeCommand += ['--jpgdir', '%s' % dstJpgDir]

        if not isShotNameBurn:
            nukeCommand += ['--shotNameBurn', 'False']
        else:
            nukeCommand += ['--shotname', rowData[Column2_org.SHOT_NAME.value]]

        cleanupEffectText = ''
        nukeEffectText = ''
        if not isEffectBurn:
            nukeCommand += ['--effectBurn', 'False']
        else:
            effectText = rowData[Column2_org.ISSUE.value]
            # clipName - clipPlateType - Effect Info
            for text in effectText.split('\n'):
                if "SpeedRamp" in text or "Retime" in text or "Scale" in text or "Rotation" in text:
                    nukeEffectText += '%s,-,%s,-,%s--' % (clipName, clipPlateType, text.replace(' ', ','))
                    cleanupEffectText += '%s - %s - %s\n' % (clipName, clipPlateType, text)

            if shotDetailDict:
                for key in sorted(shotDetailDict.keys()):
                    print shotDetailDict[key]
                    for text in shotDetailDict[key][Column2_org.ISSUE.name.lower()].split('\n'):
                        if "SpeedRamp" in text or "Retime" in text or "Scale" in text or "Rotation" in text:
                            nukeEffectText += '%s,-,%s,-,%s--' % (clipName, clipPlateType, text.replace(' ', ','))
                            cleanupEffectText += '%s - %s - %s\n' % (clipName, clipPlateType, text)

            if nukeEffectText:
                nukeCommand += ['--effect', '%s' % nukeEffectText]

        print ' '.join(nukeCommand)
        task.addCommand(author.Command(argv=nukeCommand, service=serviceKey))

        # Third Jpg To Mov
        burnInMovFileName = os.path.join(movDirectory, '_shot_burn_in_', '%s.mov' % shotName)
        movCommand = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
        movCommand += ['ffmpeg_converter', '-i', os.path.join(dstJpgDir, burnInDirName)]
        movCommand += ['-r', '%s' % '23.98']# rowData[Column2_org.MOV_CUT_FPS.value]]
        movCommand += ['-a', wavFilePath]
        movCommand += ['-o', burnInMovFileName]
        movCommand += ['-c', 'h264']
        task.addCommand(author.Command(argv=movCommand, service=serviceKey))
        if not shotMovFileDict.has_key(seq):
            shotMovFileDict[seq] = []
        shotMovFileDict[seq].append(burnInMovFileName)

        # Veloz Upload
        velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
        velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/scripts/VelozUpload.py']
        velozUploadCommand += ['--showCode', project_code]
        velozUploadCommand += ['--shotName', shotName]
        velozUploadCommand += ['--movFile', burnInMovFileName]
        velozUploadCommand += ['--description', '%s' % editDate]
        velozUploadCommand += ['--context', 'publish/edit']
        velozUploadCommand += ['--sync', str(sync)]
        task.addCommand(author.Command(argv=velozUploadCommand, service=serviceKey))

        # Shot Thumbnail
        shotThumbnailCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
        shotThumbnailCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/scripts/VelozUpload.py']
        shotThumbnailCommand += ['--showCode', project_code]
        shotThumbnailCommand += ['--shotName', shotName]
        shotThumbnailCommand += ['--movFile', os.path.join(dstJpgDir, burnInDirName, '%s.%05d.jpg' % (shotName, rowData[Column2_org.MOV_CUT_IN.value]))]
        shotThumbnailCommand += ['--description', '%s' % editDate]
        shotThumbnailCommand += ['--context', 'icon']
        task.addCommand(author.Command(argv=shotThumbnailCommand, service=serviceKey))

        # Remove JPG DIR
        removeCommand = ['/usr/bin/rm', '-rf', dstJpgDir]
        task.addCommand(author.Command(argv=removeCommand, service=serviceKey))

        # Make Tactic Sequence & Shot
        shot_data = {
            'sequence_code': seq,
            'code': shotName,
            'name': shotName,
            'status': 'Waiting',
            'pipeline_code': shot_search_type,
            'frame_in': int(rowData[Column2_org.FRAME_IN.value]),
            'frame_out': int(rowData[Column2_org.FRAME_OUT.value]),
            'edit_order': int(rowData[Column2_org.EDIT_ORDER.value]),
            'description_vfxdetail': cleanupEffectText
        }

        seqQueryData = tactic.query(sequence_search_type, filters=[('code', seq)])
        if not seqQueryData:
            tactic.insert(sequence_search_type, sequence_data)

        # But Check Tactic shot
        queryData = tactic.query(shot_search_type, filters=[('code', shotName)])

        # already shot setup
        if queryData:
            shotQueryData = queryData[0]
            curFrameIn = shotQueryData['frame_in']
            curFrameOut = shotQueryData['frame_out']
            editOrder = shotQueryData['edit_order']
            vfxdetail = shotQueryData['description_vfxdetail']

            data = {}
            editPos = ""
            editCompare = ""
            editDur = ""
            # VELOZ UPDATE LIST
            # 1. FRAME_IN, FRAME_OUT
            if curFrameIn and curFrameIn != rowData[Column2_org.FRAME_IN.value]:
                data['frame_in'] = int(rowData[Column2_org.FRAME_IN.value])
                offsetValue = curFrameIn - rowData[Column2_org.FRAME_IN.value]
                if offsetValue > 0:
                    editPos = "top"
                    editCompare = "add"
                    editDur = offsetValue
                elif offsetValue < 0:
                    editPos = "top"
                    editCompare = "delete"
                    editDur = -offsetValue
            if curFrameOut and curFrameOut != rowData[Column2_org.FRAME_OUT.value]:
                data['frame_out'] = int(rowData[Column2_org.FRAME_OUT.value])
                offsetValue = curFrameOut - rowData[Column2_org.FRAME_OUT.value]
                if offsetValue > 0:
                    editPos = "end"
                    editCompare = "delete"
                    editDur = offsetValue
                elif offsetValue < 0:
                    editPos = "end"
                    editCompare = "add"
                    editDur = -offsetValue
            if editOrder != rowData[Column2_org.EDIT_ORDER.value] and modifyEditOrder:
                data['edit_order'] = int(rowData[Column2_org.EDIT_ORDER.value])

            if vfxdetail:
                beforeDetail = vfxdetail.split('\n\n')[0]
                print beforeDetail

            # 2. STATUS
            if data != {}:
                if editCompare == "delete":
                    data['status'] = "Changed"
                elif editCompare == "add":
                    data['status'] = 'Re-Scan'

                print "Veloz Update Data :", data
                shotBuildKey = tactic.build_search_key(shot_search_type, shotName)
                print shotBuildKey
                try:
                    tactic.update(shotBuildKey, data)
                except:
                    tactic.abort()

                # 3. NOTE
                if editPos and editCompare and editDur:
                    note = FORMAT.EDIT_CHANGED_MSG.format(EDITDATE=editDate,
                                                          EDIT_POS=editPos,EDIT_COMPARE=editCompare,EDIT_DUR=int(editDur),
                                                          BEFORE_FRAME_IN_OUT='%d-%d' % (curFrameIn, curFrameOut), BEFORE_DURATION=curFrameOut-curFrameIn+1,
                                                          NEW_FRAME_IN_OUT='%d-%d' % (int(rowData[Column2_org.FRAME_IN.value]), int(rowData[Column2_org.FRAME_OUT.value])),
                                                          NEW_DURATION=int(rowData[Column2_org.FRAME_OUT.value]) - int(rowData[Column2_org.FRAME_IN.value]) + 1)
                    print shotQueryData
                    print note
                    print shot_search_type, {'search_type':'{projCode}/shot?project={projCode}'.format(projCode=project_code),
                                                     'search_id': shotQueryData['id'],
                                                     'login':login,
                                                     'context':'publish',
                                                     'process':'publish',
                                                     'note': note}
                    try:
                        tactic.insert('sthpw/note', {'search_type':'{projCode}/shot?project={projCode}'.format(projCode=project_code),
                                                         'search_id': shotQueryData['id'],
                                                         'login':login,
                                                         'context':'publish',
                                                         'process':'publish',
                                                         'note': note})
                    except:
                        tactic.abort()
        else:
            try:
                tactic.insert(shot_search_type, shot_data)
            except:
                tactic.abort()

        rootTask.addChild(task)
        shotNameList.append(shotName)
        row += 1

    tactic.finish('%s %s EditCutSpool Finish' % (showName, editDate))

    # SAVE MODIFY SCAN LISDT
    filename, ext = os.path.splitext(xlsFileName)
    editModifyFilePath = filename + '_modifyList' + ext
    editModifyListExcel.save(filename + '_modifyList' + ext)

    # Export SEQ MOV
    if uploadSeqMov:
        for seqName in shotMovFileDict.keys():
            seqMovCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.7.1', '--']
            seqMovCommand += ['rvio', '-v']
            for shotMov in sorted(shotMovFileDict[seqName]):
                seqMovCommand.append(shotMov)

            # seqName = os.path.basename(shotMovFileList[0]).split('_')[0]
            seqMov = os.path.join(os.path.dirname(xlsFileName), '%s.mov' % seqName)
            seqMovCommand += ['-o %s' % seqMov]
            rootTask.addCommand(author.Command(argv=seqMovCommand, service=serviceKey))

            # Checkin SEQ MOV
            seqVelozUploadCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
            seqVelozUploadCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/scripts/VelozUpload.py']
            seqVelozUploadCmd += ['--showCode', project_code]
            seqVelozUploadCmd += ['--seqName', seqName]
            seqVelozUploadCmd += ['--movFile', seqMov]
            seqVelozUploadCmd += ['--description', '%s' % editDate]
            seqVelozUploadCmd += ['--context', 'edit']
            seqVelozUploadCmd += ['--sync', str(sync)]
            rootTask.addCommand(author.Command(argv=seqVelozUploadCmd , service=serviceKey))

    # Notification Cmd
    notificationCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
    notificationCmd += ['BotMsg', '-r', roomIdMapper[showName.lower()], '-b', 'VelozBot', '-m', FORMAT.EDITOKMSG.format(PROJECT=showName, EDITDATE=editDate)]
    notificationCmd += ['-f', editModifyFilePath]
    rootTask.addCommand(author.Command(argv=notificationCmd, service=serviceKey))

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
