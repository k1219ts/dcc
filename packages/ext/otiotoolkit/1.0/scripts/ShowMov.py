#coding:utf-8
import os, sys, datetime, getpass

# Tractor
import tractor.api.author as author

from Define import TRACTOR, Column2
from core import excelManager

def main(xlsFileName):
    showName = xlsFileName.split('/')[3].lower() # if /prod_nass

    # Tractor Job Setup
    job = author.Job()
    job.title = '(Editorial Show MOV) %s' % os.path.basename(xlsFileName)
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

    rootTask = author.Task(title='shot setup')
    job.addChild(rootTask)

    splitXlsFile = xlsFileName.split('/')
    editDate = datetime.datetime.now().strftime('%Y/%m/%d')
    editDir = ''
    if 'prod_nas' in splitXlsFile:
        editIndex = splitXlsFile.index('edit')
        editDate = splitXlsFile[editIndex + 1].split('_')[0]
        editDir = '/'.join(splitXlsFile[:editIndex + 2])
        print editDate

    # shotMovFileDict = {}
    movFileList = []
    # seqNameList = []
    rowIndex = 1
    row = 1

    showMovDir = os.path.join(editDir, 'show_mov')
    backupMovFile = ''
    while row < excelMng.count() - 1:
        shotName = excelMng.getRow(row, Column2.SHOT_NAME.name)
        clipName = excelMng.getRow(row, Column2.CLIP_NAME.name)
        if excelMng.getRow(row, Column2.XML_NAME.name):
            movFile = excelMng.getRow(row, Column2.XML_NAME.name).replace('.xml', '.mov')
            backupMovFile = movFile
        else:
            movFile = backupMovFile
        movCutIn = int(excelMng.getRow(row, Column2.MOV_CUT_IN.name))
        # shotEditOrder = len(movFileList) + 1

        movCutDuration = int(excelMng.getRow(row + 1, Column2.MOV_CUT_IN.name))
        for tmpIdx in range(row + 1, excelMng.count()):
            nextShotName = excelMng.getRow(tmpIdx, Column2.SHOT_NAME.name)
            if shotName != nextShotName:
                movCutDuration = int(excelMng.getRow(tmpIdx, Column2.MOV_CUT_IN.name)) - movCutIn
                row = tmpIdx - 1
                break

        burninDirName = 'burnin'

        print shotName
        dataFileName = '{ROW}_{SHOTNAME}'.format(ROW=row,
                                                 SHOTNAME=shotName)

        task = author.Task(title='{DSTMOV}'.format(DSTMOV=dataFileName))
        # First EditMOV to Shot jpg
        command = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
        command += ['rvio', '-v', movFile]
        command += ['-t', '%s-%s' % (movCutIn, movCutIn + movCutDuration - 1)]
        command += ['-in709', '-outsrgb']

        dstJpgDir = os.path.join(showMovDir, '_shot_burn_in_', '%s' % dataFileName)
        dstJpgRule = os.path.join(dstJpgDir, '%s.' % dataFileName + '%06d.jpg')

        if not os.path.exists(dstJpgDir):
            os.makedirs(dstJpgDir)
        command += ['-o', '%s' % dstJpgRule]
        task.addCommand(author.Command(argv=command, service=TRACTOR.SERVICE_KEY))

        # Seconds burn in Using Nuke
        nukeCommand = ['/backstage/dcc/DCC', 'rez-env', 'nuke-12.2.4', '--']
        nukeCommand += ['nukeX', '-i', '-t', '-X', 'Write1']
        nukeCommand += [os.environ['OTIOTOOLKIT_SCRIPT_PATH'] + '/burnInUsingNuke.py']
        nukeCommand += ['--jpgdir', '%s' % dstJpgDir]
        if shotName:
            nukeCommand += ['--shotname', shotName]
        else:
            nukeCommand += ['--shotname', dataFileName]
        nukeCommand += ['--effectBurn', 'False']
        print ' '.join(nukeCommand)
        task.addCommand(author.Command(argv=nukeCommand, service=TRACTOR.SERVICE_KEY))

        # Third Jpg To Mov
        burnInMovFileName = os.path.join(showMovDir, '_shot_burn_in_', '%s.mov' % dataFileName)
        movCommand = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
        movCommand += ['ffmpeg_converter', '-i', os.path.join(dstJpgDir, burninDirName)]
        movCommand += ['-r', '%s' % '%.2f' % 23.976]
        movCommand += ['-o', burnInMovFileName]
        movCommand += ['-c', 'h264']
        task.addCommand(author.Command(argv=movCommand, service=TRACTOR.SERVICE_KEY))

        # Remove JPG DIR
        removeCommand = ['/usr/bin/rm', '-rf', dstJpgDir]
        task.addCommand(author.Command(argv=removeCommand, service=TRACTOR.SERVICE_KEY))

        movFileList.append(burnInMovFileName)

        rootTask.addChild(task)

        row += 1

    # First.5 MOV to Shot wav
    soundCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
    soundCommand += ['rvio', '-v', movFile]

    wavFilePath = os.path.join(showMovDir, '%s.wav' % movFile.replace('.mov', ''))
    if not os.path.exists(os.path.dirname(wavFilePath)):
        os.makedirs(os.path.dirname(wavFilePath))
    soundCommand += ['-o', '%s' % wavFilePath]
    rootTask.addCommand(author.Command(argv=soundCommand, service=TRACTOR.SERVICE_KEY))

    seqMovCommand = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
    seqMovCommand += ['rvio', '-v']
    showMovFile = os.path.join(showMovDir, '%s.mov' % showName)
    for movFile in movFileList:
        seqMovCommand.append(movFile)
        # seqMovCommand += ['-codec', '']
    seqMovCommand.append(wavFilePath)
    seqMovCommand += ['-o %s' % showMovFile]
    rootTask.addCommand(author.Command(argv=seqMovCommand, service=TRACTOR.SERVICE_KEY))

    author.setEngineClientParam(hostname=TRACTOR.IP, port=TRACTOR.PORT, user=getpass.getuser(), debug=True)
    # print job.as_tcl()
    job.spool()
    author.closeEngineClient()

if __name__ == "__main__":
    xlsFileName = sys.argv[-1]
    main(xlsFileName)
