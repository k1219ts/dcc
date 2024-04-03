import subprocess
import os
import xlrd2
import glob
import sys
# import DXRulebook.Interface as rb

# if not '/backstage/libs/tactic' in sys.path:
#     sys.path.append('/backstage/libs/tactic')
from tactic_client_lib import TacticServerStub

# filename = '/home/daeseok.chae/Desktop/wdl_request_20201016.xlsx'
#
# # Read shotInfo XLS
# shotXLS = xlrd2.open_workbook(filename)
# sheet = shotXLS.sheet_by_name('Sheet1')

# ocio setup
ocioConfig = os.path.join(os.getenv('REZ_OCIO_CONFIGS_ROOT'), 'config.ocio') # old style
# ocioConfig = os.getenv('OCIO') # renewal new style
oiioCmdRule = '/backstage/dcc/DCC rez-env oiio -- oiiotool {INPUT_RULE} --colorconfig {OCIO_CONFIG} --tocolorspace "out_rec709" -o {OUTPUT_RULE}'

# tactic setup
TACTIC_IP = '10.0.0.51'
login = 'daeseok.chae'
password = 'dexter#1322'
showCode = 'show106'
tactic = TacticServerStub(login=login, password=password, server=TACTIC_IP, project=showCode)

for plateType in os.listdir('/show/slc/_2d/shot/TEST/TEST_1100/plates'):
    if 'main' not in plateType:
        continue
    shotName = 'TEST_1100'
    seq = shotName.split('_')[0]
    plateDir = os.path.join('/show/slc/_2d/shot/TEST/TEST_1100/plates', plateType, 'v001')
    
    # copy plate
    # dstPlateDir = '/show/wdl/_2d/shot/{SEQ}/{SHOT}/plates/main1'.format(SEQ=seq, SHOT=shotName)
    # verList = sorted(glob.glob('%s/v*' % dstPlateDir))
    # if verList:
    #     lastVersion = os.path.basename(verList[-1])
    # else:
    #     lastVersion = 'v000'
    # verNum = int(lastVersion[1:])
    plateVer = 'v001'
    fileName = '{SHOTNAME}_{TYPE}_{VER}.{FRAME}.exr'
    startFrame = 1001
    
    dstPlateDir = os.path.join(plateDir)
    # print dstPlateDir
    # if not os.path.exists(dstPlateDir):
    #     os.makedirs(dstPlateDir)
    #
    # for index, filename in enumerate(sorted(os.listdir(plateDir))):
    #     print index, filename
    #     dstFilePath = os.path.join(dstPlateDir, fileName.format(SHOTNAME=shotName, VER=plateVer, FRAME=(startFrame + index)))
    #     srcFilePath = os.path.join(plateDir, filename)
    #     cmd = 'cp {srcFile} {dstFile}'.format(srcFile=srcFilePath, dstFile=dstFilePath)
    #     print cmd
    #     os.system(cmd)

    # copy end, process make mov
    # print dstPlateDir
    plateFileName = os.listdir(dstPlateDir)[0]
    splitFileName = plateFileName.split('.')
    splitFileName[-2] = '#'
    inputRule = os.path.join(dstPlateDir, '.'.join(splitFileName))
    print inputRule
    print

    splitFileName = plateFileName.split('.')
    splitFileName[-2] = '#'
    splitFileName[-1] = 'jpg'
    outputRule = os.path.join(dstPlateDir, 'jpg', '.'.join(splitFileName))
    print outputRule
    print

    proxyDir = os.path.dirname(outputRule)
    if not os.path.exists(proxyDir):
        os.makedirs(proxyDir)
    cmd = oiioCmdRule.format(INPUT_RULE=inputRule, OCIO_CONFIG=ocioConfig, OUTPUT_RULE=outputRule)
    print cmd
    print
    # Convert plate to JPG image
    # p = subprocess.Popen(cmd)
    # p.wait()
    os.system(cmd)

    # make mov
    movFilePath = os.path.join(os.path.dirname(dstPlateDir), '{SHOTNAME}_{TYPE}_{VER}.mov'.format(SHOTNAME=shotName, TYPE=plateType, VER=plateVer))
    movCmd = '/backstage/dcc/DCC rez-env python-2 ffmpeg_toolkit -- ffmpeg_converter -c h264 -i %s -o %s' % (proxyDir, movFilePath)
    print movCmd
    print
    # p = subprocess.Popen(movCmd)
    # p.wait()
    os.system(movCmd)

    # TACTIC UPLOAD
    search_type = '%s/shot' % showCode
    context = 'publish/edit'
    description = 'H264 Codec'
    # build tactic search key
    search_key = tactic.build_search_key(search_type, shotName)
    print search_type, search_key
    print movFilePath
    tactic.start()
    try:
        snapshot = tactic.simple_checkin(search_key, context, movFilePath, description=description, mode='copy')
    except Exception as e:
        print e.message
        tactic.abort()
    else:
        tactic.finish()

    removeCmd = 'rm -rf %s' % proxyDir
    # p = subprocess.Popen(removeCmd)
    # p.wait()
    os.system(removeCmd)
    

'''        
plateDir = '/show/slc/shot/TEST/TEST_1010/plates/main1/v01'
ocioConfig = os.path.join(os.getenv('REZ_OCIO_CONFIGS_ROOT'), 'config.ocio')

cmdRule = 'oiiotool {INPUT_RULE} --colorconfig {OCIO_CONFIG} --colorconvert "ACES - ACES2065-1" out_rec709 -o {OUTPUT_RULE}'

# ACES-2065-1 -> aces out_rec709

fileName = os.listdir(plateDir)[0]
splitFileName = fileName.split('.')
splitFileName[-2] = '#'
inputRule = os.path.join(plateDir, '.'.join(splitFileName))
print inputRule
print

fileName = os.listdir(plateDir)[0]
splitFileName = fileName.split('.')
splitFileName[-2] = '#'
splitFileName[-1] = 'jpg'
outputRule = os.path.join(plateDir, 'jpg', '.'.join(splitFileName))
print outputRule
print

outputDir = os.path.dirname(outputRule)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

cmd = cmdRule.format(INPUT_RULE=inputRule, OCIO_CONFIG=ocioConfig, OUTPUT_RULE=outputRule)
print cmd
print
# p = subprocess.Popen(cmd)
# p.wait()

# Make MOV
movCmd = 'ffmpeg_converter -i %s -o %s' % (outputDir, os.path.dirname(plateDir))
print movCmd
# p = subprocess.Popen(movCmd)
# p.wait()
'''
