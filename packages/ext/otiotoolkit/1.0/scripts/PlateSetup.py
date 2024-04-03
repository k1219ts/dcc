import xlrd2, xlwt
import datetime
import glob
import dxConfig
import requests
import Msg
import utils
import DXRulebook.Interface as rb

from Define import Column2, RescanColumn
from core.calculator import *

API_KEY = "c70181f2b648fdc2102714e8b5cb344d"

def getShowCode(showName):
    projectName = showName
    requestParam = dict() # eqaul is requestParm = {}
    requestParam['api_key'] = API_KEY
    requestParam['name'] = projectName

    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=dxConfig.getConf('TACTIC_IP')), params=requestParam).json()
    return responseData[0]['code']

def getShotInfo(showCode, shotName):
    taskshot = {}
    taskshot['api_key'] = API_KEY

    taskshot['project_code'] = showCode
    taskshot['code'] = shotName

    infos = requests.get("http://%s/dexter/search/shot.php" % (dxConfig.getConf('TACTIC_IP')),
                         params=taskshot).json()
    # print showCode, shotName, infos
    return infos

def getResolution(imgFile):
    img = oiio.ImageInput.open(imgFile)
    print img.spec().width, img.spec().height
    return '{WIDTH}x{HEIGHT}'.format(WIDTH=img.spec().width, HEIGHT=img.spec().height)

def writeXlsRow(sheet, rowIndex, rowData, st=None):
    if st == None:
        st = xlwt.easyxf('pattern: pattern solid;')
        st.pattern.pattern_fore_colour = 1
        st.pattern.pattern_back_colour = 0

    for col in range(len(rowData)):
        sheet.write(rowIndex, col, rowData[col], st)

def getFileList(searchPath, plateName):
    # print "chdir : %s" % searchPath
    # print "CMD :", 'find -L -name *%s*' % plateName
    # os.chdir(searchPath)
    files = []
    value = os.popen('find -L %s -name *%s*' % (searchPath, plateName)).read()
    tempFiles = value.strip().split('\n')
    for tmp in tempFiles:
        if '._' not in tmp:
            files.append(tmp)
    # print 'tempFiles:', tempFiles
    # length = len(tempFiles)
    return files

def isSkip(rowData, showCode, plateVersionUp):
    # Skip List
    if not rowData[Column2.SHOT_NAME.value]:
        return True, "No ShotName"

    shotInfo = getShotInfo(showCode, rowData[Column2.SHOT_NAME.value])
    if shotInfo and shotInfo[0]['status'] == 'NoVFX':
        return True, "NoVFX"

    if not rowData[Column2.TYPE.value]:
        return True, "type not exists"

    if (rowData[Column2.TC_IN.value] == '' and rowData[Column2.TC_OUT.value] == ''):
        return True, "TC_IN, TC_OUT not exists"

    if plateVersionUp == False and "delete" in rowData[Column2.EDIT_ISSUE.value] and rowData[Column2.ORIGINAL_ROOT_PATH.value]:
        return True, 'already plate setup'

    return False, ''

def getPlateVersion(showName, shotName, plateType):

    coder = rb.Coder()
    argv = coder.N.SHOTNAME.Decode(shotName)

    targetPlateDir = os.path.join('/show', showName.lower(), '_2d', 'shot', argv.seq, shotName,  'plates', plateType)
    versionList = []
    for i in glob.glob('{}/*'.format(targetPlateDir)):
        tmp = coder.D.PLATES.Decode(i)
        if tmp.ver:
            versionList.append(i)

    for version in versionList:
        if not os.path.isdir(version):
            versionList.remove(version)
    return versionList

def doIt(xlsFileName, scanRootPath='', shotNameList=None, plateVersionUp=False):
    print "Do It"
    writeXlsFileName = xlsFileName.replace('.xls', "_plate.xls")
    errorXlsFileName = xlsFileName.replace('.xls', "_error.xls")
    showName = xlsFileName.split('/')[3]
    showCode = getShowCode(showName.lower())

    # load show _config
    utils.setShowConfig(showName.lower())

    plateListExcel = xlrd2.open_workbook(xlsFileName)
    sheet = plateListExcel.sheet_by_name('scan_list')

    wBook = xlwt.Workbook(encoding='utf-8')
    wSheet = wBook.add_sheet('scan_list', cell_overwrite_ok=True)

    errorBook = xlwt.Workbook(encoding='utf-8')
    errorSheet = errorBook.add_sheet('error_list', cell_overwrite_ok=True)

    for column in RescanColumn:
        errorSheet.write(0, column.value, column.name.lower())

    errorRowIndex = 1

    excludeShotList = []
    # Column2 Setup
    writeXlsRow(wSheet, 0, sheet.row_values(0))
    for row in range(1, sheet.nrows):
        rowData = sheet.row_values(row)
        shotName = rowData[Column2.SHOT_NAME.value]
        plateType = rowData[Column2.TYPE.value]
        plateRootDir = rowData[Column2.ORIGINAL_ROOT_FOLDER.value]
        movCutIn = rowData[Column2.MOV_CUT_IN.value]
        if plateRootDir == '':
            plateRootDir = '/stuff/%s/scan' % showName.lower()
        plateFolderPath = rowData[Column2.ORIGINAL_ROOT_PATH.value]
        plateName = rowData[Column2.CLIP_NAME.value]
        plateTcIn = rowData[Column2.TC_IN.value]
        plateTcOut = rowData[Column2.TC_OUT.value]
        issue = rowData[Column2.ISSUE.value]

        isSkipVar, Reason = isSkip(rowData, showCode, plateVersionUp)
        if isSkipVar:
            print "Skip List (%s): %s" % (Reason, shotName)
            st = xlwt.easyxf('pattern: pattern solid;')
            # print xlwt.Style.colour_map
            if Reason == "NoVFX":
                st.pattern.pattern_fore_colour = xlwt.Style.colour_map['dark_red']
                st.pattern.pattern_back_colour = 0
            elif Reason == "TC_IN, TC_OUT not exists":
                st.pattern.pattern_fore_colour = xlwt.Style.colour_map['dark_green']
                st.pattern.pattern_back_colour = 0
            elif Reason == "type not exists":
                st.pattern.pattern_fore_colour = xlwt.Style.colour_map['ivory']
                st.pattern.pattern_back_colour = 0
            elif Reason == "already plate setup":
                st.pattern.pattern_fore_colour = xlwt.Style.colour_map['gray80']
                st.pattern.pattern_back_colour = 0

            writeXlsRow(wSheet, row, rowData, st)
            continue

        if shotNameList and shotName not in shotNameList:
            writeXlsRow(wSheet, row, rowData)
            continue

        FPS = rowData[Column2.SCAN_FPS.value]
        if FPS == 23.98:
            FPS = 24
        elif FPS == 29.98:
            FPS = 30
        FPS = float(FPS)

        if scanRootPath != '':
            plateRootDir = scanRootPath
            rowData[Column2.ORIGINAL_ROOT_FOLDER.value] = scanRootPath

        if plateTcIn > plateTcOut:
            temp = plateTcIn
            plateTcIn = plateTcOut
            plateTcOut = temp

        inFrames = otio.opentime.from_timecode(plateTcIn, FPS)
        endFrames = otio.opentime.from_timecode(plateTcOut, FPS)

        # print plateRootDir, plateName
        plateFiles = getFileList(plateRootDir, plateName)
        # Cleanup
        plateDict = {}
        # print plateFiles, plateRootDir, plateName
        for plateFile in plateFiles:
            filename = os.path.basename(plateFile)
            directory = os.path.dirname(plateFile)

            if directory == '':
                continue

            if not plateDict.has_key(directory):
                plateDict[directory] = list()
            plateDict[directory].append(filename)

        if plateDict == {}:
            Msg.error("Not Found Plate : %s" % shotName)
            st = xlwt.easyxf('pattern: pattern solid;')
            st.pattern.pattern_fore_colour = xlwt.Style.colour_map['light_orange']
            st.pattern.pattern_back_colour = 0
            writeXlsRow(wSheet, row, rowData, st)

            errorSheet.write(errorRowIndex, RescanColumn.SHOT_NAME.value, shotName)
            errorSheet.write(errorRowIndex, RescanColumn.CLIP_NAME.value, plateName)
            # scanDate = plateRootFolder.split('_')[0]
            errorSheet.write(errorRowIndex, RescanColumn.REQUEST_ISSUE.value, "Not find scan")
            errorSheet.write(errorRowIndex, RescanColumn.REQUEST_TC_IN.value, inFrames.to_timecode())
            errorSheet.write(errorRowIndex, RescanColumn.REQUEST_TC_OUT.value, endFrames.to_timecode())
            errorSheet.write(errorRowIndex, RescanColumn.REQUEST_DATE.value, datetime.datetime.now().strftime('%Y/%m/%d'))
            errorSheet.write(errorRowIndex, RescanColumn.MOV_CUT_IN.value, movCutIn)
            excludeShotList.append((shotName, plateType, inFrames.to_timecode(), endFrames.to_timecode()))
            errorRowIndex += 1
            continue

        for pid, plateRootFolder in enumerate(sorted(plateDict.keys(), reverse=True)):
            plateDir = os.path.join(plateRootDir, plateRootFolder)
            # plateFiles = sorted(os.listdir(plateDir))
            plateFiles = sorted(plateDict[plateRootFolder])
            # print plateFiles

            startFile = plateFiles[0]
            endFile = plateFiles[-1]

            if os.path.isdir(os.path.join(plateDir, startFile)):
                if pid == len(plateDict.keys()) - 1:
                    writeXlsRow(wSheet, row, rowData)
                continue

            tcIn = getTCInfo(os.path.join(plateDir, startFile), FPS)
            tcOut = getTCInfo(os.path.join(plateDir, endFile), FPS)

            imgTcIn = otio.opentime.from_timecode(tcIn, FPS)
            imgTcOut = otio.opentime.from_timecode(tcOut, FPS)
            imgDuration = otio.opentime.RationalTime.duration_from_start_end_time(imgTcIn, imgTcOut)
            imgRange = otio.opentime.TimeRange(imgTcIn, imgDuration)

            if tcIn == inFrames.to_timecode() and tcOut == endFrames.to_timecode(): # just TC
                if len(plateDict[plateRootFolder]) < (endFrames.to_frames() - inFrames.to_frames()):
                    writeXlsRow(wSheet, row, rowData)
                    excludeShotList.append((shotName, plateType, plateRootFolder, tcIn, inFrames.to_timecode(), tcOut, endFrames.to_timecode()))
                    continue

                inFilePath = os.path.join(plateDir, startFile)
                rowData[Column2.ORIGINAL_ROOT_PATH.value] = plateRootFolder

                versionList = getPlateVersion(showName, shotName, plateType)
                version = utils.Ver(1)
                if versionList:
                    for versionPath in versionList:
                        if os.listdir(versionPath) == 0:
                            versionList.remove(versionPath)
                    version = utils.Ver(len(versionList) + 1)
                rowData[Column2.VERSION.value] = version
                print 'inFilePath:', inFilePath
                rowData[Column2.RESOLUTION.value] = getResolution(inFilePath)

                writeXlsRow(wSheet, row, rowData)
                break
            elif tcIn <= inFrames.to_timecode() and tcOut >= endFrames.to_timecode(): # TC Range IN TC
                if len(plateDict[plateRootFolder]) < (endFrames.to_frames() - inFrames.to_frames()):
                    print "%s : %s [ %d : %d ]" % (shotName, plateRootFolder, len(plateDict[plateRootFolder]),
                                                   (endFrames.to_frames() - inFrames.to_frames()))
                    writeXlsRow(wSheet, row, rowData)
                    excludeShotList.append((shotName, plateType, plateRootFolder, tcIn, inFrames.to_timecode(), tcOut, endFrames.to_timecode()))
                    continue

                inFilePath = os.path.join(plateDir, startFile)
                rowData[Column2.ORIGINAL_ROOT_PATH.value] = plateRootFolder

                versionList = getPlateVersion(showName, shotName, plateType)
                version = utils.Ver(1)
                if versionList:
                    for versionPath in versionList:
                        if os.listdir(versionPath) == 0:
                            versionList.remove(versionPath)
                    version = utils.Ver(len(versionList) + 1)
                rowData[Column2.VERSION.value] = version
                rowData[Column2.RESOLUTION.value] = getResolution(inFilePath)

                writeXlsRow(wSheet, row, rowData)
                break
            # else:
            #     inFilePath = os.path.join(plateDir, startFile)
            #     rowData[Column2.ORIGINAL_ROOT_PATH.value] = plateRootFolder
            #
            #     versionList = getPlateVersion(showName, shotName, plateType)
            #     version = 'v001'
            #     if versionList:
            #         for versionPath in versionList:
            #             if os.listdir(versionPath) == 0:
            #                 versionList.remove(versionPath)
            #         version = 'v%03d' % (len(versionList) + 1)
            #     rowData[Column2.VERSION.value] = version
            #     rowData[Column2.RESOLUTION.value] = getResolution(inFilePath)
            #
            #     writeXlsRow(wSheet, row, rowData)
            #     break
        else:
            print shotName, inFrames.to_timecode(), endFrames.to_timecode()
            # print "HI"
            writeXlsRow(wSheet, row, rowData)
            for pid, plateRootFolder in enumerate(sorted(plateDict.keys(), reverse=True)):
                plateDir = os.path.join(plateRootDir, plateRootFolder)
                # plateFiles = sorted(os.listdir(plateDir))
                plateFiles = sorted(plateDict[plateRootFolder])
                startFile = plateFiles[0]
                endFile = plateFiles[-1]

                tcIn = getTCInfo(os.path.join(plateDir, startFile), FPS)
                tcOut = getTCInfo(os.path.join(plateDir, endFile), FPS)

                errorSheet.write(errorRowIndex, RescanColumn.SHOT_NAME.value, shotName)
                errorSheet.write(errorRowIndex, RescanColumn.CLIP_NAME.value, plateName)
                # scanDate = plateRootFolder.split('_')[0]
                splitPlateDir = plateDir.split('/')
                print 'splitPlateDir:', splitPlateDir
                scans = [word for word in splitPlateDir if 'scan' in word]
                scanIndex = splitPlateDir.index(scans[0])
                # scanIndex = splitPlateDir.index('scan')
                scanDate = splitPlateDir[scanIndex + 1]
                errorSheet.write(errorRowIndex, RescanColumn.RECEIVED_DATE.value, scanDate)
                errorSheet.write(errorRowIndex, RescanColumn.RECEIVED_FOLDER.value, plateRootFolder)
                errorSheet.write(errorRowIndex, RescanColumn.REQUEST_ISSUE.value, "mismatch TimeCode")
                errorSheet.write(errorRowIndex, RescanColumn.REQUEST_TC_IN.value, inFrames.to_timecode())
                errorSheet.write(errorRowIndex, RescanColumn.REQUEST_TC_OUT.value, endFrames.to_timecode())
                errorSheet.write(errorRowIndex, RescanColumn.REQUEST_DATE.value, datetime.datetime.now().strftime('%Y/%m/%d'))
                errorSheet.write(errorRowIndex, RescanColumn.ORIGINAL_TC_IN.value, tcIn)
                errorSheet.write(errorRowIndex, RescanColumn.ORIGINAL_TC_OUT.value, tcOut)
                try:
                    duration = otio.opentime.RationalTime.duration_from_start_end_time(otio.opentime.from_timecode(tcIn, FPS),
                                                                                       otio.opentime.from_timecode(tcOut, FPS))
                    errorSheet.write(errorRowIndex, RescanColumn.ORIGINAL_DURATION.value, duration.value)
                except:
                    duration = None
                    errorSheet.write(errorRowIndex, RescanColumn.ORIGINAL_DURATION.value, "None")
                errorSheet.write(errorRowIndex, RescanColumn.MOV_CUT_IN.value, movCutIn)
                excludeShotList.append((shotName, plateType, inFrames.to_timecode(), endFrames.to_timecode()))
                errorRowIndex += 1

    if excludeShotList:
        errorBook.save(errorXlsFileName)
    wBook.save(writeXlsFileName)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()

    # mov file name
    argparser.add_argument('-f', '--file', dest='file', type=str, required=True, help='xls file name')
    argparser.add_argument('-sc', '--scan', dest='scan', type=str, default='', help='set scan root directory')
    argparser.add_argument('-sn', '--shotName', dest='shotName', type=str, nargs='*', help='Specialize Shot Plate Setup')

    args, unknown = argparser.parse_known_args(sys.argv)

    print args.file, args.scan, args.shotName

    doIt(args.file, args.scan, args.shotName)
