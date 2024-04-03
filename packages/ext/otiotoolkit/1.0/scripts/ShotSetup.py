#coding:utf-8
import os
import getpass
import requests
import datetime
import argparse
import sys
import utils
import DXRulebook.Interface as rb

# Tractor
import tractor.api.author as author

# Tactic
from tactic_client_lib import TacticServerStub

from core import excelManager

from Define import TACTIC, TRACTOR, roomIdMapper, FORMAT, Column2
import DBConfig

def getShowCode(showName):
    '''
    input showName (emd) -> get showCode in Tactic (show106)
    :param showName: project code name
    :return: show code, blaad china sync options (True, False)
    '''
    requestParam = dict()
    requestParam['api_key'] = TACTIC.API_KEY
    requestParam['name'] = showName
    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=TACTIC.IP), params=requestParam)

    projectInfo = responseData.json()[0]
    return projectInfo['code'], projectInfo['sync']

def writeXlsRow(srcExcelMng, srcIndex, dstExcelMng, dstIndex, sheet='scan_list'):
    '''
    # XLS Row Full Write
    :param sheet:
    :param rowIndex:
    :param rowData:
    :return:
    '''
    for column in Column2:
        dstExcelMng.setRow(dstIndex, column.name, srcExcelMng.getRow(srcIndex, column.name, sheet), sheet)

def appendTransition(tactic, tacticTransitionCount, showName, editDate):
    tacticTransitionCount += 1
    if tacticTransitionCount % 100 == 0:
        tactic.finish('Shot Setup %s - %s [%d]' % (showName, editDate, tacticTransitionCount/100))
        tactic.start()
    return tactic, tacticTransitionCount

def shotSetup(xlsFileName, isShotNameBurn, isEffectBurn, uploadSeqMov, changeEditOrder, editFPS, velozUpdate, onlyShotNameList=[], velozStatus='', vfxDetailUpdate=True, velozAutoDuration=False):
    showName = xlsFileName.split('/')[3] # if /prod_nass

    # load show _config
    utils.setShowConfig(showName.lower())

    project_code, sync = getShowCode(showName.lower())
    print showName.lower(), project_code, velozUpdate
    # DB
    coll = DBConfig.db[showName.lower()]

    # Tractor Job Setup
    job = author.Job()
    job.title = '(EDITORIAL) %s' % os.path.basename(xlsFileName)
    job.comment = 'sourcefile : ' + xlsFileName
    job.service = TRACTOR.SERVICE_KEY
    job.maxactive = TRACTOR.MAX_ACTIVE
    job.tier = TRACTOR.TIER
    job.tags = [TRACTOR.TAGS]
    job.projects = [TRACTOR.PROJECT]
    job.priority = TRACTOR.PRIORITY

    if not os.path.exists(xlsFileName):
        assert False, "Not found XLS File"

    excelMng = excelManager.ExcelMng()
    excelMng.load(xlsFileName)

    scanListExcelMng = excelManager.ExcelMng()

    rootTask = author.Task(title='shot setup')
    job.addChild(rootTask)

    # Tactic Login
    if velozUpdate:
        tactic = TacticServerStub(login=TACTIC.LOGIN, password=TACTIC.PASSWORD, server=TACTIC.IP, project=project_code)
        tacticTransitionCount = 0

    splitXlsFile = xlsFileName.split('/')
    editDate = datetime.datetime.now().strftime('%Y/%m/%d')
    editDir = ''
    if 'prod_nas' in splitXlsFile:
        editIndex = splitXlsFile.index('edit')
        if splitXlsFile[editIndex + 1].split('_')[0]:
            editDate = splitXlsFile[editIndex + 1].split('_')[0]
            print editDate
        editDir = '/'.join(splitXlsFile[:editIndex + 2])

    shotMovFileDict = {}
    shotNameList = []
    seqNameList = []
    rowIndex = 1
    row = 1

    shotMovDir = os.path.join(editDir, 'shot_mov')
    wavDir = os.path.join(editDir, 'wav')
    seqMovDir = os.path.join(editDir, 'seq_mov')

    tmpShotName = 'None'
    while row < excelMng.count() - 1:
        print row
        # if excelMng.getRow(row, Column2.EDIT_ORDER.name) == "":
        #     row += 1
        #     continue
        shotName = excelMng.getRow(row, Column2.SHOT_NAME.name)
        issue = excelMng.getRow(row, Column2.ISSUE.name)
        editIssue = excelMng.getRow(row, Column2.EDIT_ISSUE.name)
        clipName = excelMng.getRow(row, Column2.CLIP_NAME.name)
        clipPlateType = excelMng.getRow(row, Column2.TYPE.name)
        startTC = excelMng.getRow(row, Column2.TC_IN.name)
        endTC = excelMng.getRow(row, Column2.TC_OUT.name)
        scanFPS = excelMng.getRow(row, Column2.SCAN_FPS.name)
        movFile = excelMng.getRow(row, Column2.XML_NAME.name).replace('.xml', '.mov')
        # frameIn = int(excelMng.getRow(row, Column2.FRAME_IN.name))
        movCutIn = int(excelMng.getRow(row, Column2.MOV_CUT_IN.name))
        # shotEditOrder = int(excelMng.getRow(row, Column2.EDIT_ORDER.name))
        shotEditOrder = len(shotNameList) + 1

        # Same Shot Check START
        if not shotName:
            row += 1
            continue

        movCutDuration = int(excelMng.getRow(row + 1, Column2.MOV_CUT_IN.name))
        for tmpIdx in range(row + 1, excelMng.count()):
            nextShotName = excelMng.getRow(tmpIdx, Column2.SHOT_NAME.name)

            print 'shotName:', shotName
            if tmpShotName not in shotName:
                tmpShotName = shotName
                print 'tmpShotName:', tmpShotName

            if tmpShotName in nextShotName:
                print 'check:', tmpShotName, nextShotName
            elif shotName != nextShotName:
                movCutDuration = int(excelMng.getRow(tmpIdx, Column2.MOV_CUT_IN.name)) - movCutIn
                break
        # print movCutIn, movCutDuration

        shot_search_type = '%s/shot' % project_code

        # if "Veloz Task" in issue:
        #     row += 1
        #     continue

        if not os.path.exists(movFile):
            movFile = movFile.replace('.mov', '.mp4')
        # movDir = os.path.dirname(movFile)

        if shotName in shotNameList:
            row += 1
            continue

        shotDetailDict = {}
        index = 0
        if row + 1 != excelMng.count() - 1:
            while row + 1 < excelMng.count() - 1 and excelMng.getRow(row + 1, Column2.SHOT_NAME.name) == shotName:
                shotDetailDict[index] = {Column2.CLIP_NAME.name.lower() : excelMng.getRow(row + 1, Column2.CLIP_NAME.name),
                                         Column2.TYPE.name.lower()      : excelMng.getRow(row + 1, Column2.TYPE.name),
                                         Column2.ISSUE.name.lower()     : excelMng.getRow(row + 1, Column2.ISSUE.name)}
                row += 1

        # Same Shot Check END

        # edit modify list up
        findItem = DBConfig.getData(coll, shotName, clipName, startTC, endTC, fps=scanFPS)
        if "top" in editIssue or "end" in editIssue or not findItem:
            writeXlsRow(excelMng, row, scanListExcelMng, rowIndex)
            rowIndex += 1

        burninDirName = 'burnin'

        coder = rb.Coder()
        argv = coder.N.SHOTNAME.Decode(shotName)

        seq = argv.seq
        sequence_search_type = '%s/sequence' % project_code
        sequence_data = {
            'code': seq,
            'name': seq,
            'original_seq_code': seq,
            'roll': 'R1',
            'status': 'Waiting',
            'pipeline_code': sequence_search_type
        }

        if velozUpdate:
            seqQueryData = tactic.query(sequence_search_type, filters=[('code', seq)])
            if not seqQueryData:
                tactic.insert(sequence_search_type, sequence_data)
                tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)

        print shotName
        task = author.Task(title='{DSTMOV}'.format(DSTMOV=shotName))
        # First EditMOV to Shot jpg
        command = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
        command += ['rvio', '-v', movFile]
        command += ['-t', '%s-%s' % (movCutIn, movCutIn + movCutDuration - 1)]
        command += ['-in709', '-outsrgb']

        # padding = len(str(movCutIn))
        # movCutEnd = movCutIn + movCutDuration - 1
        # if padding < len(str(movCutEnd)):
        #     padding = len(str(movCutEnd))

        dstJpgDir = os.path.join(shotMovDir, '_shot_burn_in_', '%s' % shotName)
        dstJpgRule = os.path.join(dstJpgDir, '%s.' % shotName + '%06d.jpg')

        if not os.path.exists(dstJpgDir):
            os.makedirs(dstJpgDir)
        command += ['-o', '%s' % dstJpgRule]
        task.addCommand(author.Command(argv=command, service=TRACTOR.SERVICE_KEY))

        # First.5 MOV to Shot wav
        soundCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
        soundCommand += ['rvio', '-v', movFile]
        soundCommand += ['-t', '%s-%s' % (movCutIn, movCutIn + movCutDuration - 1)]

        wavFilePath = os.path.join(wavDir, '%s.wav' % shotName)
        if not os.path.exists(os.path.dirname(wavFilePath)):
            os.makedirs(os.path.dirname(wavFilePath))
        soundCommand += ['-o', '%s' % wavFilePath]
        task.addCommand(author.Command(argv=soundCommand, service=TRACTOR.SERVICE_KEY))

        # Seconds burn in Using Nuke
        nukeCommand = ['/backstage/dcc/DCC', 'rez-env', 'nuke-12.2.4', '--']
        nukeCommand += ['nukeX', '-i', '-t', '-X', 'Write1']
        # nukeCommand += [os.environ['OTIOTOOLKIT_SCRIPT_PATH'] + '/burnInUsingNuke.py']
        nukeCommand += ['/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts/' + '/burnInUsingNuke.py']
        nukeCommand += ['--jpgdir', '%s' % dstJpgDir]

        if not isShotNameBurn:
            nukeCommand += ['--shotNameBurn', 'False']
        else:
            nukeCommand += ['--shotname', shotName]

        cleanupEffectText = ''
        nukeEffectText = ''
        if not isEffectBurn:
            nukeCommand += ['--effectBurn', 'False']
        else:
            effectText = issue
            # clipName - clipPlateType - Effect Info
            for text in effectText.split('\n'):
                if "SpeedRamp" in text or "Retime" in text or "Scale" in text or "Rotation" in text:
                    nukeEffectText += '%s,-,%s,-,%s--' % (clipName, clipPlateType, text.replace(' ', ','))
                    cleanupEffectText += '%s - %s - %s\n' % (clipName, clipPlateType, text)

            if shotDetailDict:
                for key in sorted(shotDetailDict.keys()):
                    sameShotClipPlateType = shotDetailDict[key][Column2.TYPE.name.lower()]
                    sameShotClipName = shotDetailDict[key][Column2.CLIP_NAME.name.lower()]
                    sameShotClipIssue = shotDetailDict[key][Column2.ISSUE.name.lower()]
                    for text in sameShotClipIssue.split('\n'):
                        if "SpeedRamp" in text or "Retime" in text or "Scale" in text or "Rotation" in text:
                            nukeEffectText += '%s,-,%s,-,%s--' % (sameShotClipName, sameShotClipPlateType, text.replace(' ', ','))
                            cleanupEffectText += '%s - %s - %s\n' % (sameShotClipName, sameShotClipPlateType, text)

            if nukeEffectText:
                nukeCommand += ['--effect', '%s' % nukeEffectText]

        print ' '.join(nukeCommand)
        task.addCommand(author.Command(argv=nukeCommand, service=TRACTOR.SERVICE_KEY))

        # Third Jpg To Mov
        burnInMovFileName = os.path.join(shotMovDir, '_shot_burn_in_', '%s.mov' % shotName)
        movCommand = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
        movCommand += ['ffmpeg_converter', '-i', os.path.join(dstJpgDir, burninDirName)]
        movCommand += ['-r', '%s' % '%.2f' % editFPS]
        movCommand += ['-a', wavFilePath]
        movCommand += ['-o', burnInMovFileName]
        movCommand += ['-c', 'h264']
        task.addCommand(author.Command(argv=movCommand, service=TRACTOR.SERVICE_KEY))

        # Shot Thumbnail
        shotThumbnailCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
        # shotThumbnailCommand += ['python', os.environ['OTIOTOOLKIT_SCRIPT_PATH'] + '/VelozUpload.py']
        shotThumbnailCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts' + '/VelozUpload.py']
        shotThumbnailCommand += ['--showCode', project_code]
        shotThumbnailCommand += ['--shotName', shotName]
        shotThumbnailCommand += ['--movFile', os.path.join(dstJpgDir, burninDirName,
                                                           '%s.%s.jpg' % (shotName, str(movCutIn).zfill(6)))]
        shotThumbnailCommand += ['--description', '%s' % editDate]
        shotThumbnailCommand += ['--context', 'icon']
        if velozUpdate:
            task.addCommand(author.Command(argv=shotThumbnailCommand, service=TRACTOR.SERVICE_KEY))

        # Remove JPG DIR
        removeCommand = ['/usr/bin/rm', '-rf', dstJpgDir]
        task.addCommand(author.Command(argv=removeCommand, service=TRACTOR.SERVICE_KEY))

        if not shotMovFileDict.has_key(seq):
            shotMovFileDict[seq] = []
        shotMovFileDict[seq].append(burnInMovFileName)

        if 'Dissolve' in issue:
            splitIssue = issue.split('\n')
            top_dissolve = 0.0
            end_dissolve = 0.0
            for sIssue in splitIssue:
                if 'Dissolve' in sIssue:
                    splitDissolve = sIssue.split('|')
                    top_dissolve = float(splitDissolve[-2].split(':')[-1])
                    end_dissolve = float(splitDissolve[-1].split(':')[-1])
                    break

            movCutDuration += end_dissolve
            movCutEnd = movCutIn + movCutDuration - 1

            command = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
            command += ['rvio', '-v', movFile]
            command += ['-t', '%s-%s' % (movCutIn, movCutEnd)]
            command += ['-in709', '-outsrgb']

            # padding = len(str(movCutIn))
            #
            # if padding < len(str(movCutEnd)):
            #     padding = len(str(movCutEnd))

            dstJpgDir = os.path.join(shotMovDir, '_shot_burn_in_', '%s_dissolve' % shotName)
            dstJpgRule = os.path.join(dstJpgDir, '%s.' % shotName + '%06d.jpg')

            if not os.path.exists(dstJpgDir):
                os.makedirs(dstJpgDir)
            command += ['-o', '%s' % dstJpgRule]
            task.addCommand(author.Command(argv=command, service=TRACTOR.SERVICE_KEY))

            # First.5 MOV to Shot wav
            soundCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
            soundCommand += ['rvio', '-v', movFile]
            soundCommand += ['-t', '%s-%s' % (movCutIn, movCutIn + movCutDuration - 1)]

            wavFilePath = os.path.join(wavDir, '%s_dissolve.wav' % shotName)
            if not os.path.exists(os.path.dirname(wavFilePath)):
                os.makedirs(os.path.dirname(wavFilePath))
            soundCommand += ['-o', '%s' % wavFilePath]
            task.addCommand(author.Command(argv=soundCommand, service=TRACTOR.SERVICE_KEY))

            # Seconds burn in Using Nuke
            nukeCommand = ['/backstage/dcc/DCC', 'rez-env', 'nuke-12.2.4', '--']
            nukeCommand += ['nukeX', '-i', '-t', '-X', 'Write1']
            # nukeCommand += [os.environ['OTIOTOOLKIT_SCRIPT_PATH'] + '/burnInUsingNuke.py']
            nukeCommand += ['/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts' + '/burnInUsingNuke.py']
            nukeCommand += ['--jpgdir', '%s' % dstJpgDir]

            if not isShotNameBurn:
                nukeCommand += ['--shotNameBurn', 'False']
            else:
                nukeCommand += ['--shotname', shotName]

            cleanupEffectText = ''
            nukeEffectText = ''
            if not isEffectBurn:
                nukeCommand += ['--effectBurn', 'False']
            else:
                effectText = issue
                # clipName - clipPlateType - Effect Info
                for text in effectText.split('\n'):
                    if "SpeedRamp" in text or "Retime" in text or "Scale" in text or "Rotation" in text:
                        nukeEffectText += '%s,-,%s,-,%s--' % (clipName, clipPlateType, text.replace(' ', ','))
                        cleanupEffectText += '%s - %s - %s\n' % (clipName, clipPlateType, text)

                if shotDetailDict:
                    for key in sorted(shotDetailDict.keys()):
                        sameShotClipPlateType = shotDetailDict[key][Column2.TYPE.name.lower()]
                        sameShotClipName = shotDetailDict[key][Column2.CLIP_NAME.name.lower()]
                        sameShotClipIssue = shotDetailDict[key][Column2.ISSUE.name.lower()]
                        for text in sameShotClipIssue.split('\n'):
                            if "SpeedRamp" in text or "Retime" in text or "Scale" in text or "Rotation" in text:
                                nukeEffectText += '%s,-,%s,-,%s--' % (
                                sameShotClipName, sameShotClipPlateType, text.replace(' ', ','))
                                cleanupEffectText += '%s - %s - %s\n' % (sameShotClipName, sameShotClipPlateType, text)

                if nukeEffectText:
                    nukeCommand += ['--effect', '%s' % nukeEffectText]

            print ' '.join(nukeCommand)
            task.addCommand(author.Command(argv=nukeCommand, service=TRACTOR.SERVICE_KEY))

            # Third Jpg To Mov
            burnInMovFileName = os.path.join(shotMovDir, '_shot_burn_in_', '%s_dissolve.mov' % shotName)
            movCommand = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
            movCommand += ['ffmpeg_converter', '-i', os.path.join(dstJpgDir, burninDirName)]
            movCommand += ['-r', '%s' % '%.2f' % editFPS]
            movCommand += ['-a', wavFilePath]
            movCommand += ['-o', burnInMovFileName]
            movCommand += ['-c', 'h264']
            task.addCommand(author.Command(argv=movCommand, service=TRACTOR.SERVICE_KEY))

            # Remove JPG DIR
            removeCommand = ['/usr/bin/rm', '-rf', dstJpgDir]
            task.addCommand(author.Command(argv=removeCommand, service=TRACTOR.SERVICE_KEY))

        # Veloz Upload
        velozUploadCommand = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
        # velozUploadCommand += ['python', os.environ['OTIOTOOLKIT_SCRIPT_PATH'] + '/VelozUpload.py']
        velozUploadCommand += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts' + '/VelozUpload.py']
        velozUploadCommand += ['--showCode', project_code]
        velozUploadCommand += ['--shotName', shotName]
        velozUploadCommand += ['--movFile', burnInMovFileName]
        velozUploadCommand += ['--description', '%s' % editDate]
        velozUploadCommand += ['--context', 'publish/edit']
        velozUploadCommand += ['--sync', str(sync)]
        if velozUpdate:
            task.addCommand(author.Command(argv=velozUploadCommand, service=TRACTOR.SERVICE_KEY))

        if onlyShotNameList and shotName not in onlyShotNameList:
            row += 1
            if not seq in seqNameList:
                shotMovFileDict.pop(seq)
            continue

        if not seq in seqNameList:
            seqNameList.append(seq)

        rootTask.addChild(task)
        shotNameList.append(shotName)

        # TACTIC SHOT SETUP
        frameOut = int(1001 + movCutDuration - 1)
        shot_data = {
            'sequence_code': seq,
            'code': shotName,
            'name': shotName,
            'status': 'Waiting',
            'pipeline_code': shot_search_type,
            'frame_in': 1001,
            'frame_out': frameOut,
            'frame_in_ani': 1001,
            'frame_out_ani': frameOut,
            'edit_order': int(shotEditOrder),
            'description_vfxdetail': cleanupEffectText
        }

        if not vfxDetailUpdate:
            shot_data['description_vfxdetail'] = ''

        # But Check Tactic shot
        if velozUpdate:
            queryData = tactic.query(shot_search_type, filters=[('code', shotName)])
            if queryData:
                shotQueryData = queryData[0]
                velozFrameIn = shotQueryData['frame_in']
                velozFrameInAni = shotQueryData['frame_in_ani']
                if velozFrameInAni == None:
                    velozFrameInAni = velozFrameIn
                velozFrameOut = shotQueryData['frame_out']
                editOrder = shotQueryData['edit_order']
                vfxdetail = shotQueryData['description_vfxdetail']

                data = {}
                editPos = ""
                editCompare = ""
                editDur = ""

                # VELOZ UPDATE LIST
                # 1. FRAME_IN, FRAME_OUT
                parseEditIssue = editIssue.split('\n')
                note = FORMAT.EDIT_CHANGED_TITLE_NOTE.format(EDITDATE=editDate)
                applyNote = False
                editDurModify = False
                for cutEditIssue in parseEditIssue:
                    print cutEditIssue
                    if cutEditIssue.strip() == '':
                        continue

                    if 'end' in cutEditIssue:
                        endCompare = cutEditIssue.split(' ')[-2]
                        endDur = cutEditIssue.split(' ')[-1]
                        if endCompare == "delete":
                            data['frame_out'] = velozFrameOut - int(endDur)
                            data['frame_out_ani'] = velozFrameOut - int(endDur)
                            if data.has_key('status') and data['status'] == 'Re-Scan':
                                data['status'] = "Re-Scan"
                            else:
                                data['status'] = "Changed"
                        elif endCompare == "add":
                            data['frame_out'] = velozFrameOut + int(endDur)
                            data['frame_out_ani'] = velozFrameOut + int(endDur)
                            data['status'] = 'Re-Scan'
                        note += cutEditIssue + '\n'
                        applyNote = True
                        editDurModify = True

                    elif 'top' in cutEditIssue:
                        topCompare = cutEditIssue.split(' ')[-2]
                        topDur = cutEditIssue.split(' ')[-1]
                        if topCompare == "delete":
                            data['frame_in'] = velozFrameIn + int(topDur)
                            data['frame_in_ani'] = velozFrameInAni + int(topDur)
                            if data.has_key('status') and data['status'] == 'Re-Scan':
                                data['status'] = "Re-Scan"
                            else:
                                data['status'] = "Changed"
                        elif topCompare == "add":
                            data['frame_in'] = velozFrameIn - int(topDur)
                            data['frame_in_ani'] = velozFrameInAni - int(topDur)
                            data['status'] = 'Re-Scan'
                        note += cutEditIssue + '\n'
                        applyNote = True
                        editDurModify = True
                    else:
                        note += cutEditIssue + '\n'
                        applyNote = True

                if editDurModify:
                    frameIn = velozFrameIn
                    if data.has_key('frame_in'):
                        frameIn = data['frame_in']

                    frameOut = velozFrameOut
                    if data.has_key('frame_out'):
                        frameOut = data['frame_out']
                    note += FORMAT.EDIT_DURATION_CHANGE_NOTE.format(BEFORE_FRAME_IN_OUT='%d-%d' % (velozFrameIn, velozFrameOut),
                                                          BEFORE_DURATION=velozFrameOut - velozFrameIn + 1,
                                                          NEW_FRAME_IN_OUT='%d-%d' % (frameIn, frameOut),
                                                          NEW_DURATION=frameOut - frameIn + 1)

                if editOrder != shotEditOrder and changeEditOrder:
                    data['edit_order'] = shotEditOrder

                if vfxdetail and vfxDetailUpdate:
                    beforeDetail = vfxdetail.split('\n\n')[0]
                    if beforeDetail != issue:
                        data['description_vfxdetail'] = issue + '\n\n' + shotQueryData['description_vfxdetail']

                velozShotDuration = velozFrameOut - velozFrameIn + 1
                if velozAutoDuration and velozShotDuration != movCutDuration and not (data.has_key('frame_in') or data.has_key('frame_out')):
                    data['frame_in'] = velozFrameIn
                    data['frame_in_ani'] = velozFrameInAni
                    data['frame_out'] = velozFrameIn + movCutDuration - 1
                    data['frame_out_ani'] = velozFrameIn + movCutDuration - 1
                    print "HI"

                # 2. STATUS
                if data != {}:
                    print "Veloz Update Data :", data
                    shotBuildKey = tactic.build_search_key(shot_search_type, shotName)
                    print shotBuildKey
                    try:
                        tactic.update(shotBuildKey, data)
                        tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
                    except:
                        tactic.abort()

                # 3. NOTE
                if applyNote:
                    print shotQueryData
                    print note
                    print shot_search_type, {'search_type':'{projCode}/shot?project={projCode}'.format(projCode=project_code),
                                                     'search_id': shotQueryData['id'],
                                                     'login':TACTIC.LOGIN,
                                                     'context':'publish',
                                                     'process':'publish',
                                                     'note': note}
                    try:
                        tactic.insert('sthpw/note', {'search_type':'{projCode}/shot?project={projCode}'.format(projCode=project_code),
                                                         'search_id': shotQueryData['id'],
                                                         'login':TACTIC.LOGIN,
                                                         'context':'publish',
                                                         'process':'publish',
                                                         'note': note})
                        tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
                    except:
                        tactic.abort()
            else:
                try:
                    tactic.insert(shot_search_type, shot_data)
                    tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
                except:
                    tactic.abort()

        row += 1

    if velozUpdate:
        for i in range(1, excelMng.count('omit_list')):
            shotName = excelMng.getRow(i, Column2.SHOT_NAME.name, 'omit_list')
            editIssue = excelMng.getRow(i, Column2.EDIT_ISSUE.name, 'omit_list')
            shot_search_type = '%s/shot' % project_code
            queryData = tactic.query(shot_search_type, filters=[('code', shotName)])
            if queryData:
                writeXlsRow(excelMng, i, scanListExcelMng, i, sheet='omit_list')
                shotQueryData = queryData[0]
                data = {}

                splitEditIssueList = editIssue.split('\n')
                note = FORMAT.EDIT_CHANGED_TITLE_NOTE.format(EDITDATE=editDate)

                data['status'] = velozStatus
                # for splitEditIssue in splitEditIssueList:
                #     # if 'VFX Status' in splitEditIssue:
                #     #     note += splitEditIssue.split(':')[-1] + '\n'
                #     #     data['status'] = splitEditIssue.split(':')[-1]
                #     # else:
                #     note += splitEditIssue

                shotBuildKey = tactic.build_search_key(shot_search_type, shotName)
                try:
                    tactic.update(shotBuildKey, data)
                    tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
                except:
                    tactic.abort()
                try:
                    tactic.insert('sthpw/note',
                                  {'search_type': '{projCode}/shot?project={projCode}'.format(projCode=project_code),
                                   'search_id': shotQueryData['id'],
                                   'login': TACTIC.LOGIN,
                                   'context': 'publish',
                                   'process': 'publish',
                                   'note': note})
                    tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
                except:
                    tactic.abort()

    if velozUpdate:
        tactic.finish('Shot Setup %s - %s' % (showName, editDate))

    # SAVE MODIFY SCAN LIST
    filename, ext = os.path.splitext(xlsFileName)
    editModifyFilePath = filename + '_modifyList' + ext
    scanListExcelMng.save(editModifyFilePath)

    # Export SEQ MOV
    if uploadSeqMov:
        if not os.path.exists(seqMovDir):
            os.makedirs(seqMovDir)

        for seqName in shotMovFileDict.keys():
            seqMovCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
            seqMovCommand += ['rvio', '-v']
            for shotMov in shotMovFileDict[seqName]:
                seqMovCommand.append(shotMov)

            # seqName = os.path.basename(shotMovFileList[0]).split('_')[0]
            seqMov = os.path.join(seqMovDir, '%s.mov' % seqName)
            # seqMovCommand += ['-codec', '']
            seqMovCommand += ['-o %s' % seqMov]
            rootTask.addCommand(author.Command(argv=seqMovCommand, service=TRACTOR.SERVICE_KEY))

            if velozUpdate:
                # Checkin SEQ MOV
                seqVelozUploadCmd = ['/backstage/dcc/DCC', 'rez-env', 'python-2', 'baselib-2.5', '--']
                # seqVelozUploadCmd += ['python', os.environ['OTIOTOOLKIT_SCRIPT_PATH'] + '/VelozUpload.py']
                seqVelozUploadCmd += ['python', '/backstage/dcc/packages/ext/otiotoolkit/1.0/scripts' + '/VelozUpload.py']
                seqVelozUploadCmd += ['--showCode', project_code]
                seqVelozUploadCmd += ['--seqName', seqName]
                seqVelozUploadCmd += ['--movFile', seqMov]
                seqVelozUploadCmd += ['--description', '%s' % editDate]
                seqVelozUploadCmd += ['--context', 'edit']
                seqVelozUploadCmd += ['--sync', str(sync)]

                rootTask.addCommand(author.Command(argv=seqVelozUploadCmd, service=TRACTOR.SERVICE_KEY))

    # Notification Cmd
    if velozUpdate:
        try:
            notificationCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
            notificationCmd += ['BotMsg', '-r', roomIdMapper[showName.lower()], '-b', 'VelozBot', '-m',
                                FORMAT.EDITOKMSG.format(PROJECT=showName, EDITDATE=editDate)]
            notificationCmd += ['-f', editModifyFilePath]
            rootTask.addCommand(author.Command(argv=notificationCmd, service=TRACTOR.SERVICE_KEY))
        except:
            pass

    author.setEngineClientParam(hostname=TRACTOR.IP, port=TRACTOR.PORT, user=getpass.getuser(), debug=True)
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
    argparser.add_argument('-nv', '--noVeloz', dest='noVeloz', type=str, help='do not action veloz update')
    argparser.add_argument('-eo', '--editOrder', dest='editOrder', type=str, help='how to changed edit order?')
    argparser.add_argument('-ef', '--editFPS', dest='editFPS', default=23.98, type=float, help='how to changed edit order?')
    argparser.add_argument('-sn', '--shotName', dest='shotName', type=str, nargs='*', default=[], help='')
    argparser.add_argument('-vss', '--velozStatus', dest='velozStatus', type=str, choices=['Omit', 'Hold', 'Waiting'], help='omit_list veloz status change')
    argparser.add_argument('-vd', '--vfxDetail', dest='vfxDetail', type=str, help='vfx detail update')
    argparser.add_argument('-vad', '--velozAutoDuration', dest='velozAutoDuration', type=str, help='Veloz Cut Duration Auto Update')

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

    noVeloz = False
    if args.noVeloz == "True":
        noVeloz = True

    vfxDetail = True
    if args.vfxDetail == "False":
        vfxDetail = False

    velozAutoDuration = False
    if args.velozAutoDuration == "True":
        velozAutoDuration = True

    shotSetup(args.file, shotNameBurn, effectBurn, velozSeqMov, editOrder, args.editFPS, noVeloz, args.shotName, args.velozStatus, vfxDetail, velozAutoDuration)
