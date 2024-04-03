#coding:utf-8
import xlrd2
import xlwt
import getpass
import glob
import requests
import sys
import argparse
import utils
import DXRulebook.Interface as rb

# Tractor
import tractor.api.author as author

# Tactic
API_KEY = "c70181f2b648fdc2102714e8b5cb344d"

from Define import *
from core.calculator import *

def getShowCode(showName):
    projectName = showName
    requestParam = dict() # eqaul is requestParm = {}
    requestParam['api_key'] = API_KEY
    requestParam['name'] = projectName
    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP="10.0.0.51"), params=requestParam)

    projectInfo = responseData.json()[0]
    return projectInfo['code']

def writeXlsRow(sheet, rowIndex, rowData):
    for col in range(len(rowData)):
        sheet.write(rowIndex, col, rowData[col])

def spool(xlsFileName, shotNameList=[], inOffset=0, outOffset=0):
    # Tractor
    TRACTOR_IP = '10.0.0.25'
    PORT = 80
    serviceKey = "Editorial"
    project = "export"
    tier = "cache"
    tags = ""

    showName = xlsFileName.split('/')[3]

    # load show _config
    utils.setShowConfig(showName.lower())

    project_code = getShowCode(showName.lower())

    print showName.lower(), project_code

    if showName.lower() == "cdh":
        showName = 'CDH1'

    # Job Setup
    job = author.Job()
    job.title = '(%s PLATE SETUP) %s' % (showName, os.path.basename(xlsFileName))
    job.comment = 'sourcefile : ' + xlsFileName
    job.service = TRACTOR.SERVICE_KEY
    job.maxactive = TRACTOR.MAX_ACTIVE
    job.tier = TRACTOR.TIER
    job.tags = [TRACTOR.TAGS]
    job.projects = [TRACTOR.PROJECT]
    job.priority = TRACTOR.PRIORITY

    if not os.path.exists(xlsFileName):
        assert False, "Not found XLS File"

    plateListExcel = xlrd2.open_workbook(xlsFileName)
    sheet = plateListExcel.sheet_by_name('scan_list')

    rootTask = author.Task(title='Shot Cutout')
    job.addChild(rootTask)

    # shotNameList = sys.argv[1:]

    # inOffset = 24
    # outOffset = 24
    # inOffset = 0
    # outOffset = 0
    print shotNameList

    plateScanListExcel = xlwt.Workbook(encoding='utf-8')
    plateScanExcelSheet = plateScanListExcel.add_sheet('scan_list')
    writeXlsRow(plateScanExcelSheet, 0, sheet.row_values(0))

    errorListExcel = xlwt.Workbook(encoding='utf-8')
    errorSheet = errorListExcel.add_sheet('scan_list')
    errorSheet.write(0, 0, 'shotName')
    errorSheet.write(0, 1, 'type')
    errorSheet.write(0, 2, 'scan_file_tcIn')
    errorSheet.write(0, 3, 'editorial_tcIn')
    errorSheet.write(0, 4, 'scan_file_tcOut')
    errorSheet.write(0, 5, 'editorial_tcOut')
    errorIndex = 1
    rowIndex = 1
    for row in range(1, sheet.nrows):
        rowData = sheet.row_values(row)
        shotName = rowData[Column2.SHOT_NAME.value]
        type = rowData[Column2.TYPE.value]
        version = rowData[Column2.VERSION.value]
        rootFolder = rowData[Column2.ORIGINAL_ROOT_FOLDER.value]
        rootPath = rowData[Column2.ORIGINAL_ROOT_PATH.value]
        exrFileName = rowData[Column2.CLIP_NAME.value]
        TC_IN = rowData[Column2.TC_IN.value]
        TC_OUT = rowData[Column2.TC_OUT.value]
        RESOLUTION = rowData[Column2.RESOLUTION.value]
        FPS = rowData[Column2.SCAN_FPS.value]

        if shotName in shotNameList:
            print shotName, rootPath, type, version

        if not shotName or not rootPath or not type or not version:
            continue

        if shotNameList and not shotName in shotNameList:
            print shotName, type
            continue

        print FPS

        if FPS == 23.98:
            FPS = 24
        elif FPS == 29.98:
            FPS = 30

        FPS = float(FPS)

        FRAME_IN = int(rowData[Column2.FRAME_IN.value])

        task = author.Task(title='{DSTMOV}'.format(DSTMOV=shotName))

        # EXR FILE IN Check
        # try:
        tcInTime = otio.opentime.RationalTime.from_timecode(TC_IN, FPS)
        tcOutTime = otio.opentime.RationalTime.from_timecode(TC_OUT, FPS)
        # except:
        #     tcInTime = otio.opentime.RationalTime.from_timecode(TC_IN, 30)
        #     tcOutTime = otio.opentime.RationalTime.from_timecode(TC_OUT, 30)
        #     fps = 30

        if tcInTime > tcOutTime:
            print "Reverse"
            temp = tcInTime
            tcInTime = tcOutTime
            tcOutTime = temp

        coder = rb.Coder()
        argv = coder.N.SHOTNAME.Decode(shotName)

        platePath = os.path.join(rootFolder, rootPath, exrFileName)
        targetFileName = os.path.join('/show', showName.lower(), '_2d', 'shot', argv.seq, shotName, 'plates', type, version)

        print targetFileName
        if os.path.exists(targetFileName):
            if len(os.listdir(targetFileName)) != 0:
                print "# Already Setup"
                continue

        if not os.path.exists(os.path.join(rootFolder, rootPath)):
            print "# Not Exists :", os.path.join(rootFolder, rootPath)
            continue

        fileExt = os.listdir(os.path.join(rootFolder, rootPath))[0].split('.')[-1]
        if fileExt not in ['exr', 'dpx']:
            fileExt = 'exr'
        padding = os.listdir(os.path.join(rootFolder, rootPath))[0].split('.')[-2]
        paddingLength = int(len(padding))
        realFileName = '%s.%s.%s' % (platePath, str(tcInTime.to_frames()).zfill(paddingLength), fileExt)

        # First TC Match in Filename
        if os.path.exists(realFileName):
            print "Match File and TC"
            print realFileName, "->", (targetFileName + '/{SHOTNAME}_{TYPE}_{VERSION}.{FRAME}.{EXT}'.format(SHOTNAME=shotName,
                                                                                                          TYPE=type,
                                                                                                          VERSION=version,
                                                                                                          FRAME=FRAME_IN,
                                                                                                          EXT=fileExt))
            startFrame = tcInTime.to_frames()
            inputFileRule = realFileName.replace(str(int(startFrame)).zfill(paddingLength), '%0' + str(paddingLength) + 'd')
            outputFileRule = os.path.join(targetFileName, '{SHOTNAME}_{TYPE}_{VERSION}.%04d.{EXT}'.format(SHOTNAME=shotName,
                                                                                                          TYPE=type,
                                                                                                          VERSION=version,
                                                                                                          EXT=fileExt))
            duration = otio.opentime.RationalTime.duration_from_start_end_time(tcInTime, tcOutTime).to_frames()

            if not os.path.exists(os.path.dirname(outputFileRule)):
                os.makedirs(os.path.dirname(outputFileRule))

            # Copy Plate
            plateCopyCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
            plateCopyCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateCopy.py']
            plateCopyCmd += ['--srcStartFrame', str(int(startFrame) - inOffset)]
            plateCopyCmd += ['--srcFileRule', '%s' % inputFileRule]
            plateCopyCmd += ['--dstFileRule', '%s' % outputFileRule]
            plateCopyCmd += ['--duration', str(duration + inOffset + outOffset)]
            plateCopyCmd += ['--frameIn', str(FRAME_IN - inOffset)]
            print ' '.join(plateCopyCmd)
            task.addCommand(author.Command(argv=plateCopyCmd, service=serviceKey))

            if version == 'v001':
                # Make Plate Mov
                plateMovCmd = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg-4.2.0', 'nuke-12.2.4', '--', 'nukeX']
                plateMovCmd += ['-i', '-t', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateMov.py']
                plateMovCmd += ['--exrDir', targetFileName]
                plateMovCmd += ['--reelname', exrFileName]
                if RESOLUTION:
                    plateMovCmd += ['--resolution', RESOLUTION]
                plateMovCmd += ['--stamp', 'stamp_noncrop']
                print ' '.join(plateMovCmd)
                task.addCommand(author.Command(argv=plateMovCmd, service=serviceKey))

                # Plate Mov Veloz Update
                velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
                velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/VelozUpload.py']
                velozUploadCommand += ['--showCode', project_code]
                velozUploadCommand += ['--shotName', shotName]
                movFileName = os.path.join(os.path.dirname(targetFileName), '{SHOTNAME}_{TYPE}_{VERSION}.mov'.format(SHOTNAME=shotName,
                                                                                                                     TYPE=type,
                                                                                                                     VERSION=version))
                velozUploadCommand += ['--movFile', movFileName]
                velozUploadCommand += ['--description', '"%s %s"' % (type, exrFileName)]
                velozUploadCommand += ['--context', 'publish/plate']
                task.addCommand(author.Command(argv=velozUploadCommand, service=serviceKey))

                # Plate mov remove
                removeCommand = ['/usr/bin/rm', '-vf', movFileName]
                task.addCommand(author.Command(argv=removeCommand, service=serviceKey))

            rootTask.addChild(task)
            writeXlsRow(plateScanExcelSheet, rowIndex, rowData)
            rowIndex += 1
            continue

        fileList = sorted(glob.glob('%s*' % platePath))
        print shotName, platePath
        startFile = fileList[0]
        endFile = fileList[-1]

        fileBaseName = os.path.basename(startFile).split('.')[0]
        platePath = os.path.join(rootFolder, rootPath, fileBaseName)

        # if match duration
        tcIn = getTCInfo(startFile, FPS)
        tcOut = getTCInfo(endFile, FPS)

        imgTcInTime = otio.opentime.from_timecode(tcIn, FPS)
        imgTcOutTime = otio.opentime.from_timecode(tcOut, FPS)
        print tcIn, tcOut, tcInTime.to_timecode(), tcOutTime.to_timecode()
        if tcIn == tcInTime.to_timecode() and tcOut == tcOutTime.to_timecode():
            print "Match File(0, -1) and TC"
            print startFile, '->', (targetFileName + '/{SHOTNAME}_{TYPE}_{VERSION}.{FRAME}.{EXT}'.format(SHOTNAME=shotName,
                                                                                                          TYPE=type,
                                                                                                          VERSION=version,
                                                                                                          FRAME=FRAME_IN,
                                                                                                          EXT=fileExt))
            startFrame = startFile.split('.')[-2]
            realFileName = '%s.%s.%s' % (platePath, str(int(startFrame)).zfill(paddingLength), fileExt)
            inputFileRule = realFileName.replace(str(int(startFrame)).zfill(paddingLength), '%0' + str(paddingLength) + 'd')
            outputFileRule = os.path.join(targetFileName, '{SHOTNAME}_{TYPE}_{VERSION}.%04d.{EXT}'.format(SHOTNAME=shotName,
                                                                                                        TYPE=type,
                                                                                                        VERSION=version,
                                                                                                        EXT=fileExt))
            duration = otio.opentime.RationalTime.duration_from_start_end_time(tcInTime, tcOutTime).to_frames()

            if not os.path.exists(os.path.dirname(outputFileRule)):
                os.makedirs(os.path.dirname(outputFileRule))

            # Copy Plate
            plateCopyCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
            plateCopyCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateCopy.py']
            plateCopyCmd += ['--srcStartFrame', str(int(startFrame) - inOffset)]
            plateCopyCmd += ['--srcFileRule', '%s' % inputFileRule]
            plateCopyCmd += ['--dstFileRule', '%s' % outputFileRule]
            plateCopyCmd += ['--duration', str(duration + inOffset + outOffset)]
            plateCopyCmd += ['--frameIn', str(FRAME_IN - inOffset)]
            print ' '.join(plateCopyCmd)
            task.addCommand(author.Command(argv=plateCopyCmd, service=serviceKey))

            if version == 'v001':
                # Make Plate Mov
                plateMovCmd = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg-4.2.0', 'nuke-12.2.4', '--', 'nukeX']
                plateMovCmd += ['-i', '-t', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateMov.py']
                plateMovCmd += ['--exrDir', targetFileName]
                plateMovCmd += ['--reelname', exrFileName]
                if RESOLUTION:
                    plateMovCmd += ['--resolution', RESOLUTION]
                plateMovCmd += ['--stamp', 'stamp_noncrop']
                print ' '.join(plateMovCmd)
                task.addCommand(author.Command(argv=plateMovCmd, service=serviceKey))

                # Plate Mov Veloz Update
                velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
                velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/VelozUpload.py']
                velozUploadCommand += ['--showCode', project_code]
                velozUploadCommand += ['--shotName', shotName]
                movFileName = os.path.join(os.path.dirname(targetFileName), '{SHOTNAME}_{TYPE}_{VERSION}.mov'.format(SHOTNAME=shotName,
                                                                                                                     TYPE=type,
                                                                                                                     VERSION=version))
                velozUploadCommand += ['--movFile', movFileName]
                velozUploadCommand += ['--description', '"%s %s"' % (type, exrFileName)]
                velozUploadCommand += ['--context', 'publish/plate']
                task.addCommand(author.Command(argv=velozUploadCommand, service=serviceKey))

                # Plate mov remove
                removeCommand = ['/usr/bin/rm', '-vf', movFileName]
                task.addCommand(author.Command(argv=removeCommand, service=serviceKey))

            rootTask.addChild(task)
            writeXlsRow(plateScanExcelSheet, rowIndex, rowData)
            rowIndex += 1
            continue

        if tcIn <= tcInTime.to_timecode() and tcOut >= tcOutTime.to_timecode():
            for exrFile in fileList:
                newTcIn = getTCInfo(exrFile, FPS)
                # print newTcIn, tcInTime.to_timecode()
                if newTcIn == tcInTime.to_timecode():
                    startFile = exrFile
                    break
            print "Match SCAN >= TC <= SCAN"
            print startFile, '->', (targetFileName + '/{SHOTNAME}_{TYPE}_{VERSION}.{FRAME}.{EXT}'.format(SHOTNAME=shotName,
                                                                                                       TYPE=type,
                                                                                                       VERSION=version,
                                                                                                       FRAME=FRAME_IN,
                                                                                                       EXT=fileExt))
            startFrame = startFile.split('.')[-2]
            realFileName = '%s.%s.%s' % (platePath, str(int(startFrame)).zfill(paddingLength), fileExt)
            inputFileRule = realFileName.replace(str(int(startFrame)).zfill(paddingLength), '%0' + str(paddingLength) + 'd')
            outputFileRule = os.path.join(targetFileName, '{SHOTNAME}_{TYPE}_{VERSION}.%04d.{EXT}'.format(SHOTNAME=shotName,
                                                                                                        TYPE=type,
                                                                                                        VERSION=version,
                                                                                                        EXT=fileExt))
            duration = otio.opentime.RationalTime.duration_from_start_end_time(tcInTime, tcOutTime).to_frames()

            if not os.path.exists(os.path.dirname(outputFileRule)):
                print os.path.dirname(outputFileRule)
                os.makedirs(os.path.dirname(outputFileRule))

            # Copy Plate
            plateCopyCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
            plateCopyCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateCopy.py']
            plateCopyCmd += ['--srcStartFrame', str(int(startFrame) - inOffset)]
            plateCopyCmd += ['--srcFileRule', '%s' % inputFileRule]
            plateCopyCmd += ['--dstFileRule', '%s' % outputFileRule]
            plateCopyCmd += ['--duration', str(duration + inOffset + outOffset)]
            plateCopyCmd += ['--frameIn', str(FRAME_IN - inOffset)]
            print ' '.join(plateCopyCmd)
            task.addCommand(author.Command(argv=plateCopyCmd, service=serviceKey))

            if version == utils.Ver(1):
                # Make Plate Mov
                plateMovCmd = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg-4.2.0', 'nuke-12.2.4', '--', 'nukeX']
                plateMovCmd += ['-i', '-t', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateMov.py']
                plateMovCmd += ['--exrDir', targetFileName]
                if RESOLUTION:
                    plateMovCmd += ['--resolution', RESOLUTION]
                plateMovCmd += ['--reelname', exrFileName]
                plateMovCmd += ['--stamp', 'stamp_noncrop']
                print ' '.join(plateMovCmd)
                task.addCommand(author.Command(argv=plateMovCmd, service=serviceKey))

                # Plate Mov Veloz Update
                velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
                velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/VelozUpload.py']
                velozUploadCommand += ['--showCode', project_code]
                velozUploadCommand += ['--shotName', shotName]
                movFileName = os.path.join(os.path.dirname(targetFileName), '{SHOTNAME}_{TYPE}_{VERSION}.mov'.format(SHOTNAME=shotName,
                                                                                                                     TYPE=type,
                                                                                                                     VERSION=version))
                velozUploadCommand += ['--movFile', movFileName]
                velozUploadCommand += ['--description', '"%s %s"' % (type, exrFileName)]
                velozUploadCommand += ['--context', 'publish/plate']
                task.addCommand(author.Command(argv=velozUploadCommand, service=serviceKey))

                # Plate mov remove
                removeCommand = ['/usr/bin/rm', '-vf', movFileName]
                task.addCommand(author.Command(argv=removeCommand, service=serviceKey))

            rootTask.addChild(task)
            writeXlsRow(plateScanExcelSheet, rowIndex, rowData)
            rowIndex += 1
            continue
        # elif (imgTcInTime and imgTcOutTime):
        #     for exrFile in fileList:
        #         newTcIn = getTCInfo(exrFile, FPS)
        #         # print newTcIn, tcInTime.to_timecode()
        #         if newTcIn == tcInTime.to_timecode():
        #             startFile = exrFile
        #             break
        #     print "Match SCAN >= TC <= SCAN"
        #     print startFile, '->', (targetFileName + '/{SHOTNAME}_{TYPE}_{VERSION}.{FRAME}.exr'.format(SHOTNAME=shotName,
        #                                                                                                TYPE=type,
        #                                                                                                VERSION=version,
        #                                                                                                FRAME=FRAME_IN))
        #     startFrame = startFile.split('.')[-2]
        #     # realFileName = '%s.%08d.exr' % (platePath, int(startFrame))
        #     # inputFileRule = startFile.replace(str('%08d' % int(startFrame)), '%08d')
        #     realFileName = '%s.%s.exr' % (platePath, str(int(startFrame)).zfill(paddingLength))
        #     inputFileRule = realFileName.replace(str(int(startFrame)).zfill(paddingLength), '%0' + str(paddingLength) + 'd')
        #     outputFileRule = os.path.join(targetFileName, '{SHOTNAME}_{TYPE}_{VERSION}.%04d.exr'.format(SHOTNAME=shotName,
        #                                                                                                 TYPE=type,
        #                                                                                                 VERSION=version))
        #     duration = otio.opentime.RationalTime.duration_from_start_end_time(tcInTime, tcOutTime).to_frames()
        #
        #     if not os.path.exists(os.path.dirname(outputFileRule)):
        #         print os.path.dirname(outputFileRule)
        #         os.makedirs(os.path.dirname(outputFileRule))
        #
        #     # Copy Plate
        #     plateCopyCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
        #     plateCopyCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateCopy.py']
        #     plateCopyCmd += ['--srcStartFrame', str(int(startFrame) - inOffset)]
        #     plateCopyCmd += ['--srcFileRule', '%s' % inputFileRule]
        #     plateCopyCmd += ['--dstFileRule', '%s' % outputFileRule]
        #     plateCopyCmd += ['--duration', str(duration + inOffset + outOffset)]
        #     plateCopyCmd += ['--frameIn', str(FRAME_IN - inOffset)]
        #     print ' '.join(plateCopyCmd)
        #     task.addCommand(author.Command(argv=plateCopyCmd, service=serviceKey))
        #
        #     # Make Plate Mov
        #     plateMovCmd = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg-4.2.0', 'nuke-12.2.4', '--', 'nukeX']
        #     plateMovCmd += ['-i', '-t', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateMov.py']
        #     plateMovCmd += ['--exrDir', targetFileName]
        #     if RESOLUTION:
        #         plateMovCmd += ['--resolution', RESOLUTION]
        #     plateMovCmd += ['--reelname', exrFileName]
        #     plateMovCmd += ['--stamp', 'stamp_noncrop']
        #     print ' '.join(plateMovCmd)
        #     task.addCommand(author.Command(argv=plateMovCmd, service=serviceKey))
        #
        #     # Plate Mov Veloz Update
        #     velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
        #     velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/VelozUpload.py']
        #     velozUploadCommand += ['--showCode', project_code]
        #     velozUploadCommand += ['--shotName', shotName]
        #     movFileName = os.path.join(os.path.dirname(targetFileName), '{SHOTNAME}_{TYPE}_{VERSION}.mov'.format(SHOTNAME=shotName,
        #                                                                                                          TYPE=type,
        #                                                                                                          VERSION=version))
        #     velozUploadCommand += ['--movFile', movFileName]
        #     velozUploadCommand += ['--description', '"%s %s"' % (type, exrFileName)]
        #     velozUploadCommand += ['--context', 'publish/plate']
        #     task.addCommand(author.Command(argv=velozUploadCommand, service=serviceKey))
        #
        #     # Plate mov remove
        #     removeCommand = ['/usr/bin/rm', '-vf', movFileName]
        #     task.addCommand(author.Command(argv=removeCommand, service=serviceKey))
        #
        #     rootTask.addChild(task)
        #     writeXlsRow(plateScanExcelSheet, rowIndex, rowData)
        #     rowIndex += 1
        #     continue
        else: # Everything Export
            print "OK"
            startFrame = startFile.split('.')[-2]
            realFileName = '%s.%s.%s' % (platePath, str(int(startFrame)).zfill(paddingLength), fileExt)
            inputFileRule = realFileName.replace(str(int(startFrame)).zfill(paddingLength), '%0' + str(paddingLength) + 'd')
            outputFileRule = os.path.join(targetFileName, '{SHOTNAME}_{TYPE}_{VERSION}.%04d.{EXT}'.format(SHOTNAME=shotName,
                                                                                                        TYPE=type,
                                                                                                        VERSION=version,
                                                                                                        EXT=fileExt))
            duration = otio.opentime.RationalTime.duration_from_start_end_time(tcInTime, tcOutTime).to_frames()

            if not os.path.exists(os.path.dirname(outputFileRule)):
                print os.path.dirname(outputFileRule)
                os.makedirs(os.path.dirname(outputFileRule))

            # Copy Plate
            plateCopyCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
            plateCopyCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateCopy.py']
            plateCopyCmd += ['--srcStartFrame', str(int(startFrame) - inOffset)]
            plateCopyCmd += ['--srcFileRule', '%s' % inputFileRule]
            plateCopyCmd += ['--dstFileRule', '%s' % outputFileRule]
            plateCopyCmd += ['--duration', str(duration + inOffset + outOffset)]
            plateCopyCmd += ['--frameIn', str(FRAME_IN - inOffset)]
            print ' '.join(plateCopyCmd)
            task.addCommand(author.Command(argv=plateCopyCmd, service=serviceKey))

            if version == utils.Ver(1):
                # Make Plate Mov
                plateMovCmd = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg-4.2.0', 'nuke-12.2.4', '--', 'nukeX']
                plateMovCmd += ['-i', '-t', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/PlateMov.py']
                plateMovCmd += ['--exrDir', targetFileName]
                if RESOLUTION:
                    plateMovCmd += ['--resolution', RESOLUTION]
                plateMovCmd += ['--reelname', exrFileName]
                plateMovCmd += ['--stamp', 'stamp_noncrop']
                print ' '.join(plateMovCmd)
                task.addCommand(author.Command(argv=plateMovCmd, service=serviceKey))

                # Plate Mov Veloz Update
                velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
                velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/VelozUpload.py']
                velozUploadCommand += ['--showCode', project_code]
                velozUploadCommand += ['--shotName', shotName]
                movFileName = os.path.join(os.path.dirname(targetFileName),
                                           '{SHOTNAME}_{TYPE}_{VERSION}.mov'.format(SHOTNAME=shotName,
                                                                                    TYPE=type,
                                                                                    VERSION=version))
                velozUploadCommand += ['--movFile', movFileName]
                velozUploadCommand += ['--description', '"%s %s"' % (type, exrFileName)]
                velozUploadCommand += ['--context', 'publish/plate']
                task.addCommand(author.Command(argv=velozUploadCommand, service=serviceKey))

                # Plate mov remove
                removeCommand = ['/usr/bin/rm', '-vf', movFileName]
                task.addCommand(author.Command(argv=removeCommand, service=serviceKey))

            rootTask.addChild(task)
            writeXlsRow(plateScanExcelSheet, rowIndex, rowData)
            rowIndex += 1
            continue

        # tcIn <= tcInTime.to_timecode() and tcOut >= tcOutTime.to_timecode():
        # print tcIn, tcInTime.to_timecode()
        # print tcOut, tcOutTime.to_timecode()
        errorSheet.write(errorIndex, 0, shotName)
        errorSheet.write(errorIndex, 1, type)
        errorSheet.write(errorIndex, 2, tcIn)
        errorSheet.write(errorIndex, 3, tcInTime.to_timecode())
        errorSheet.write(errorIndex, 4, tcOut)
        errorSheet.write(errorIndex, 5, tcOutTime.to_timecode())
        errorIndex += 1
        # assert False, "ERROR ! %s - %s" % (shotName, type)

    # SAVE PLATE SCAN LISDT
    filename, ext = os.path.splitext(xlsFileName)
    plateScaleFilePath = filename + '_scanList' + ext
    plateScanListExcel.save(plateScaleFilePath)

    if errorIndex != 1:
        errorFilePath = filename + '_errorList' + ext
        errorListExcel.save(errorFilePath)

    # Notification Cmd
    try:
        notificationCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
        notificationCmd += ['BotMsg', '-r', roomIdMapper[showName.lower()], '-b', 'VelozBot', '-m', FORMAT.PLATEOKMSG.format(PROJECT=showName)]
        notificationCmd += ['-f', plateScaleFilePath]
        rootTask.addCommand(author.Command(argv=notificationCmd, service=serviceKey))
    except:
        print 'BotMsg Error!!'

    author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user=getpass.getuser(), debug=True)
    # print job.as_tcl()
    job.spool()
    author.closeEngineClient()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # mov file name
    argparser.add_argument('-f', '--file', dest='file', type=str, required=True, help='')
    argparser.add_argument('-sn', '--shotName', dest='shotName', type=str, nargs='*', default=[], help='')
    argparser.add_argument('-ios', '--inOffset', dest='inOffset', type=int, default=0, help='')
    argparser.add_argument('-oos', '--outOffset', dest='outOffset', type=int, default=0, help='')

    args, unknown = argparser.parse_known_args(sys.argv)

    spool(args.file, args.shotName, args.inOffset, args.outOffset)
