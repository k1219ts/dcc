import xlrd2, xlwt
import datetime
import glob
import dxConfig
import requests
import Msg

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
    # print "CMD :", 'find -name %s*' % plateName
    os.chdir(searchPath)
    value = os.popen('find -name %s*' % plateName).read()
    tempFiles = value.strip().split('\n')
    # print tempFiles
    # length = len(tempFiles)
    return tempFiles

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
    targetPlateDir = os.path.join('/show', showName.lower(), '_2d', 'shot', shotName.split('_')[0], shotName, 'plates', plateType)
    versionList = glob.glob('{}/v*'.format(targetPlateDir))
    for version in versionList:
        if not os.path.isdir(version):
            versionList.remove(version)
    return versionList

def doIt(xlsFileName):
    print "Do It"
    errorXlsFileName = xlsFileName.replace('.xls', "_error.xls")
    showName = xlsFileName.split('/')[3].lower()

    plateListExcel = xlrd2.open_workbook(xlsFileName)
    sheet = plateListExcel.sheet_by_name('scan_list')

    errorBook = xlwt.Workbook(encoding='utf-8')
    errorSheet = errorBook.add_sheet('error_list', cell_overwrite_ok=True)

    errorRowIndex = 1

    # Column2 Setup
    writeXlsRow(errorSheet, 0, sheet.row_values(0))
    for row in range(1, sheet.nrows):
        rowData = sheet.row_values(row)
        shotName = rowData[Column2.SHOT_NAME.value]
        plateType = rowData[Column2.TYPE.value]
        plateTcIn = rowData[Column2.TC_IN.value]
        plateTcOut = rowData[Column2.TC_OUT.value]
        plateFPS = rowData[Column2.SCAN_FPS.value]
        editIssue = rowData[Column2.EDIT_ISSUE.value]

        versionList = sorted(getPlateVersion(showName, shotName, plateType))
        if versionList and versionList[-1]:
            plateFiles = sorted(glob.glob("%s/*.exr" % versionList[-1]))
            if not plateFiles:
                plateFiles = sorted(glob.glob("%s/*.dpx" % versionList[-1]))
            imgStartTC = getTCInfo(plateFiles[0], plateFPS)
            imgEndTC = getTCInfo(plateFiles[-1], plateFPS)

            if plateTcIn == imgStartTC and plateTcOut == imgEndTC:
                continue
            elif plateTcIn == imgEndTC and plateTcOut == imgStartTC:
                continue
            # elif 'delete' in editIssue:
            #     continue
            else:
                rowData[-1] = '%s-%s' % (imgStartTC, imgEndTC)
                writeXlsRow(errorSheet, errorRowIndex, rowData)
                errorRowIndex += 1

        # if row == 10:
        #     break

    if errorRowIndex > 1:
        errorBook.save(errorXlsFileName)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()

    # mov file name
    argparser.add_argument('-f', '--file', dest='file', type=str, required=True, help='xls file name')

    args, unknown = argparser.parse_known_args(sys.argv)

    print args.file

    doIt(args.file)
