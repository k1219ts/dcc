#coding:utf-8
import xlrd2
import os
from Define import *
import opentimelineio as otio

def getFileList(searchPath, clipName):
    # print "CMD :", 'find -name %s.*' % plateName
    os.chdir(searchPath)
    value = os.popen('find -name %s.*' % clipName).read()
    tempFiles = value.strip().split('\n')
    # length = len(tempFiles)
    return tempFiles

xlsFileName = '/prod_nas/__DD_PROD/PRAT2/edit/201130/_xls_to_cgsup/PRAT2_rescan_list_20201203_re.xlsx'
dirPath = os.path.dirname(xlsFileName)
garbage, ext= os.path.splitext(xlsFileName)
print ext

if not os.path.exists(xlsFileName):
    assert False, "Not found XLS File"

rescanExcel = xlrd2.open_workbook(xlsFileName)
sheet = rescanExcel.sheet_by_name('rescan_list')

xmlFile = xlsFileName.replace(ext, '.xml')
edlFile = xlsFileName.replace(ext, '.edl')
movFile = xlsFileName.replace(ext, '.mov')

# make stuff/prat2/stuff/onset data
timeline = otio.schema.Timeline()
clipTrack = otio.schema.Track(name='Clip Track')
timeline.tracks.append(clipTrack)

onsetRoot = '/stuff/prat2/stuff/onset'

targetMovList = []
for row in range(1, sheet.nrows):
    rowData = sheet.row_values(row)

    sourceMov = getFileList(onsetRoot, rowData[RescanColumn.CLIP_NAME.value])

    if not sourceMov:
        assert False, 'not found onset mov'

    # mediaReference = otio.schema.ExternalReference(available_range=otio.opentime.TimeRange(self.otioData.global_start_time, self.otioData.duration()),
    #     target_url=self.movFilePath)
    startTime = otio.opentime.RationalTime.from_timecode(rowData[RescanColumn.REQUEST_TC_IN.value], 24.0)
    endTime = otio.opentime.RationalTime.from_timecode(rowData[RescanColumn.REQUEST_TC_OUT.value], 24.0)
    duration = otio.opentime.RationalTime.duration_from_start_end_time(startTime, endTime)

    available_startTime = otio.opentime.RationalTime.from_timecode(rowData[RescanColumn.ORIGINAL_TC_IN.value], 24.0)
    available_duration = otio.opentime.RationalTime(int(rowData[RescanColumn.ORIGINAL_DURATION.value]), 24.0)

    mediaReference = otio.schema.ExternalReference(target_url=os.path.join(onsetRoot, sourceMov[0]),
                                                   available_range=otio.opentime.TimeRange(available_startTime, available_duration))

    movClip = otio.schema.Clip(name=rowData[RescanColumn.CLIP_NAME.value],
                               source_range=otio.opentime.TimeRange(startTime, duration),
                               media_reference=mediaReference)
    clipTrack.append(movClip)

    # Make diectory
    if not os.path.exists(os.path.join(dirPath, '_DI_mov_')):
        os.makedirs(os.path.join(dirPath, '_DI_mov_'))

    sourceMovPath = os.path.join(onsetRoot, sourceMov[0])
    targetMovPath = os.path.join(dirPath, '_DI_mov_', os.path.basename(sourceMov[0]))
    targetMovList.append(targetMovPath)
    print 'sourceMovPath', sourceMovPath
    print 'targetMovPath', targetMovPath

    cmd = '/backstage/dcc/DCC rez-env rv-7.7.1 -- rvio -v %s -t %s-%s -o %s' % (sourceMovPath, int(startTime.value), int(endTime.value), targetMovPath)
    print cmd
    os.system(cmd)

# Export SEQ MOV
combineMovCmd = '/backstage/dcc/DCC rez-env rv-7.7.1 -- rvio -v '
for targetMov in targetMovList:
    combineMovCmd += '%s ' % targetMov
combineMovCmd += '-o %s' % movFile
os.system(combineMovCmd)

os.system('rm -rf %s' % os.path.join(dirPath, '_DI_mov_'))

otio.adapters.write_to_file(timeline, xmlFile)
otio.adapters.write_to_file(timeline, edlFile)