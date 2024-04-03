#coding:utf-8
import opentimelineio as otio
import os, subprocess, pprint, getpass
from Define import Column2, FORMAT, CLIPTYPE, TRACTOR
import Msg
from core.FlatternTrack import FlatternTrack
from core.excelManager import ExcelMng
import core.calculator as calc
from core.listMessageBox import QListMessageBox
from core import calculator
import DBConfig
# Tractor
import tractor.api.author as author
from PySide2 import QtWidgets

class FCPXML7Parser():
    def __init__(self, xmlFilePath, showName='', plateType='main'):
        self.editFilePath = xmlFilePath
        filename, extension = os.path.splitext(self.editFilePath)
        self.filename = os.path.basename(filename)
        self.dirpath = os.path.dirname(xmlFilePath)

        splitXlsFile = xmlFilePath.split('/')
        editDir = ''
        if 'prod_nas' in splitXlsFile:
            editIndex = splitXlsFile.index('edit')
            editDir = '/'.join(splitXlsFile[:editIndex + 2])
        else:
            assert False, "not find edit directory"

        self.editDir = editDir
        self.excelFilePath = os.path.join(self.editDir, 'editorial', self.filename + '.xls')
        self.inspectOtioFile = os.path.join(self.editDir, 'editorial', self.filename + '.otio')
        print editDir, self.excelFilePath, self.inspectOtioFile
        self.movFilePath = filename + '.mov'
        self.plateType = plateType

        self.showName = showName
        self.showName = self.getShowName()
        self.coll = DBConfig.db[self.showName]

        # Check Edit Changed
        editFileList = DBConfig.getEditFileList(self.coll)
        self.diffEditFileList = []
        if editFileList:
            box = QListMessageBox(reversed(editFileList))
            box.listWidget.setCurrentRow(0)
            box.exec_()
            if box.result == True:
                # self.diffEditFileList = box.listWidget.selectedItems()
                for item in box.listWidget.selectedItems():
                    self.diffEditFileList.append(item.text())

        self.openFileData()
        self.globalStartFrame = self.getGlobalStartFrame()
        self.globalEndFrame = self.otioData.global_start_time.to_frames() + self.otioData.duration().to_frames()
        self.editTimeline = FlatternTrack(self.otioData, self.movFilePath)

    def getShowName(self):
        if self.showName:
            return self.showName
        showName = calculator.parseShowName(self.editFilePath)
        return showName.lower()

    def openFileData(self):
        '''
        write xls file & read xml file
        :return:
        '''
        self.excelMng = ExcelMng()
        self.otioData = otio.adapters.read_from_file(self.editFilePath, 'fcp_xml')

        if not os.path.exists(self.movFilePath):
            self.movFilePath = self.movFilePath.replace('.mov', '.mp4')
            if not os.path.exists(self.movFilePath):
                Msg.ERROR('not found movFilePath')
                assert False, 'not found movFilePath'

        # Check MOV TimeCode
        movMetadata = calc.getMovMetadata(self.movFilePath)
        isSetMovTC = False
        movTC = '00:00:00:00'
        try:
            movTC = movMetadata['streams'][-1]['tags']['timecode']
        except Exception as e:
            isSetMovTC = True
            pprint.pprint(movMetadata)

        if isSetMovTC == False and movTC != self.otioData.global_start_time.to_timecode():
            isSetMovTC = True

        if isSetMovTC:
            # Rename MOV FILE
            backupFile = self.movFilePath.replace('.mov', '_TCError.mov')
            os.rename(self.movFilePath, backupFile)
            cmd = '/backstage/dcc/DCC rez-env ffmpeg_toolkit -- ffmpeg -i {INPUT} -map 0 -map -0:d -c copy -timecode {TIMECODE} {OUTPUT}'.format(INPUT=backupFile,
                                                                                                                                                 OUTPUT=self.movFilePath,
                                                                                                                                                 TIMECODE=self.otioData.global_start_time.to_timecode())
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while p.poll() == None:
                output = p.stdout.readline()
                if output:
                    print output.strip()

    def getGlobalStartFrame(self):
        frame = self.otioData.global_start_time.to_frames()
        if len(str(frame)) >= 7:
            return otio.opentime.RationalTime.from_timecode(self.otioData.global_start_time.to_timecode(), self.otioData.global_start_time.rate).to_frames()
        try:
            return self.otioData.global_start_time.to_frames()
        except:
            return 0

    def doIt(self):
        print "doIt"
        editCutInFrameList = self.editTimeline.getEditCutFrameList()
        cutIndex = 0
        rowIndex = 1
        editOrder = 1

        editList = []
        self.omitList = []
        if self.diffEditFileList:
            print self.diffEditFileList
            for diffEditFile in self.diffEditFileList:
                print diffEditFile
                editList += DBConfig.editList(self.coll, diffEditFile)

        while cutIndex < len(editCutInFrameList):
            clipDataList = self.editTimeline.getClipList(editCutInFrameList[cutIndex])
            plateIndex = 0
            for clipIndex, clip in enumerate(clipDataList):
                row = rowIndex

                if clip.isSlate():
                    editOrder -= 1
                    continue

                # if clip.isMissingReference():  # source file missing
                #     print editOrder, clip.schemaName
                #     editOrder -= 1
                #     break
                # if clip.schemaName == 'Clip' and clip.isTextClip():
                #     continue

                newEditOrder = editOrder

                try:
                    clipName = clip.getClipName()
                    fpsBase = clip.getFPS()
                    startTime = clip.getStartTime().to_timecode()
                    endTime = clip.getEndTime().to_timecode()
                    durationTime = clip.getDurationTime()
                except:
                    "### pass ERROR:", clipName
                    continue
                # scanDuration = otio.opentime.RationalTime.duration_from_start_end_time(
                #     otio.opentime.from_timecode(startTime, fpsBase),
                #     otio.opentime.from_timecode(endTime, fpsBase)).value

                clipTimeRange = clip.getTimeRange()
                clipRetime = clip.getEffect("Retime")
                editTCRange = clipTimeRange
                issue = ''
                editIssue = ''

                try:
                    preOrderClip = self.editTimeline.getClipList(editCutInFrameList[cutIndex - 1])[clipIndex]
                    preOrderClipName = preOrderClip.getClipName()
                    preOrderClipRetime = preOrderClip.getEffect("Retime")
                except:
                    preOrderClip = None
                    preOrderClipName = ''
                    preOrderClipRetime = None

                excelIndex = row
                if preOrderClipName == clipName and (clipRetime or preOrderClipRetime):
                    # Check Reverse
                    clipReverse = False
                    if clipRetime and clipRetime['reverse']:
                        clipReverse = True

                    preOrderClipReverse = False
                    if preOrderClipRetime and preOrderClipRetime['reverse']:
                        preOrderClipReverse = True

                    # print editOrder, "Same Clip", (clip.getStartTime() - preOrderClip.getEndTime()).value
                    # print clip.getStartTime().to_timecode(), preOrderClip.getEndTime().to_timecode()
                    if (clipRetime or preOrderClipRetime) and clipReverse == preOrderClipReverse and (clip.getStartTime() - preOrderClip.getEndTime()).value < 5:
                        minTrack = len(self.editTimeline.getClipList(editCutInFrameList[cutIndex - 1]))
                        if minTrack > len(self.editTimeline.getClipList(editCutInFrameList[cutIndex])):
                            minTrack = len(self.editTimeline.getClipList(editCutInFrameList[cutIndex]))

                        isSameClipList = True
                        # print minTrack, len(self.editTimeline.getClipList(editCutInFrameList[cutIndex - 1])), len(self.editTimeline.getClipList(editCutInFrameList[cutIndex]))
                        for index in range(minTrack):
                            aClip = self.editTimeline.getClipList(editCutInFrameList[cutIndex])[index]
                            bClip = self.editTimeline.getClipList(editCutInFrameList[cutIndex - 1])[index]
                            if not aClip.isSpeedRamp(bClip):
                                isSameClipList = False

                        if isSameClipList:
                            # print "# Merge", preOrderClip.getEndTime(), clip.getStartTime(), (clip.getStartTime() - preOrderClip.getEndTime()).value
                            excelIndex = row - len(self.editTimeline.getClipList(editCutInFrameList[cutIndex - 1]))

                if excelIndex != row:
                    # Merge List : issue, shot duration, scan duration, frame out, mov cut duration
                    # Overwrite List : End TC,

                    # merge
                    issue = self.excelMng.getRow(excelIndex, Column2.ISSUE.name) + '\n'
                    editIssue = self.excelMng.getRow(excelIndex, Column2.EDIT_ISSUE.name) + '\n'
                    shotDuration = int(self.excelMng.getRow(excelIndex, Column2.CLIP_DURATION.name))
                    shotDuration += clip.noOpticalDuration.value
                    if "Retime" in issue:
                        splitIssueList = issue.split('\n')
                        for index, splitIssue in enumerate(splitIssueList):
                            if 'Retime' in splitIssue:
                                retimeSpeed = splitIssue.split(' ')[-1][:-1]
                                speedRamp = FORMAT.SPEEDRAMPTC.format(RETIME=retimeSpeed,
                                                                      TC_IN=preOrderClip.getStartTime().to_timecode(),
                                                                      TC_OUT=preOrderClip.getEndTime().to_timecode())
                                splitIssueList[index] = speedRamp

                        issue = '\n'.join(splitIssueList)
                        # issue += '\n'

                    # overwrite
                    self.excelMng.setRow(excelIndex, Column2.TC_OUT.name, endTime)

                if clip.issue:
                    issue += clip.issue
                else:
                    if clipRetime:
                        if 'SpeedRamp' in issue:
                            issue += FORMAT.SPEEDRAMPTC.format(RETIME=clipRetime['speed'],
                                                               TC_IN=startTime,
                                                               TC_OUT=endTime)
                        else:
                            issue += FORMAT.RETIME.format(RETIME=clipRetime['speed'])
                    if clip.getEffect("Scale"):
                        issue += FORMAT.SCALE.format(SCALE=clip.getEffect("Scale"))
                    if clip.getEffect("Rotation"):
                        issue += FORMAT.ROTATION.format(ROTATE=clip.getEffect("Rotation"))

                plateType = '%s%d' % (self.plateType, plateIndex + 1)
                frameIn = 1001 # + ((clip.getFrame() + self.globalStartFrame) - editCutInFrameList[cutIndex])
                frameOut = 1001 + clip.noOpticalDuration.value
                shotDuration = clip.noOpticalDuration.value
                shotName = ''
                movCutIn = editCutInFrameList[cutIndex] # - clip.dissolve['in_offset'].value

                if clip.IsRetime():
                    plateType += "_org"

                # Make Excel Manager
                if excelIndex != row:
                    # already excel data
                    newEditOrder = self.excelMng.getRow(excelIndex, Column2.EDIT_ORDER.name)
                    startTime = self.excelMng.getRow(excelIndex, Column2.TC_IN.name)
                    shotDuration = int(self.excelMng.getRow(excelIndex, Column2.CLIP_DURATION.name)) + clip.noOpticalDuration.value
                    frameOut = int(self.excelMng.getRow(excelIndex, Column2.FRAME_OUT.name)) + clip.noOpticalDuration.value + 1
                    shotName = self.excelMng.getRow(excelIndex, Column2.SHOT_NAME.name)
                    plateType = self.excelMng.getRow(excelIndex, Column2.TYPE.name)
                    movCutIn = self.excelMng.getRow(excelIndex, Column2.MOV_CUT_IN.name)

                    rowIndex -= 1

                # Get DB Data
                findItem = DBConfig.getData(self.coll, '', clipName, startTime, endTime, fps=fpsBase)

                isRescan = False
                if findItem:
                    # print findItem
                    beforeIssue = findItem[Column2.ISSUE.name.lower()].replace('&', '\n')
                    if "Retime" in beforeIssue or clip.getEffect('Retime'):
                        splitBeforeIssue = beforeIssue.split('\n')
                        clipSpeed = 0
                        if clip.getEffect('Retime'):
                            clipSpeed = int(float(clip.getEffect('Retime')["speed"]))

                        isEditRetimeChange = False
                        beforeRetimeSpeed = 0
                        for sIssueItem in splitBeforeIssue:
                            if "Retime" in sIssueItem:
                                beforeRetimeSpeed = int(float(sIssueItem.split(' ')[-1][:-1]))

                                if beforeRetimeSpeed != clipSpeed:
                                    editIssue += FORMAT.EDIT_RETIME_CHANGED_MSG.format(BEFORE_RETIME=beforeRetimeSpeed,
                                                                                       AFTER_RETIME=clipSpeed)
                                    isEditRetimeChange = True
                                    break

                        if not isEditRetimeChange and clipSpeed != beforeRetimeSpeed:
                            beforeRetimeSpeed = 0
                            editIssue += FORMAT.EDIT_RETIME_CHANGED_MSG.format(BEFORE_RETIME=beforeRetimeSpeed,
                                                                               AFTER_RETIME=clip.getEffect('Retime')[
                                                                                   'speed'])

                    if clip.getClipType() == CLIPTYPE.CLIP:
                        if startTime != findItem[Column2.TC_IN.name.lower()]:
                            # Msg.warning("startTC :", startTime, "alreadyTC_IN :", findItem[Column2.TC_IN.name.lower()])

                            originalTC_IN = otio.opentime.RationalTime.from_timecode(findItem[Column2.TC_IN.name.lower()], fpsBase)
                            offset = originalTC_IN - editTCRange.start_time
                            frameIn = findItem[Column2.FRAME_IN.name.lower()]
                            frameIn -= offset.value
                            if offset.value <= 0: # nag value
                                editIssue += "top delete %d\n" % -offset.value
                            else:
                                editIssue += "top add %d\n" % offset.value
                                isRescan = True
                        if endTime != findItem[Column2.TC_OUT.name.lower()]:
                            # Msg.warning("endTC :", endTime, "alreadyTC_OUT :", findItem[Column2.TC_OUT.name.lower()])
                            originalTC_OUT = otio.opentime.RationalTime.from_timecode(findItem[Column2.TC_OUT.name.lower()], fpsBase)
                            offset = originalTC_OUT - otio.opentime.RationalTime.from_timecode(endTime, fpsBase)
                            if offset.value <= 0: # nag value
                                editIssue += "end add %d\n" % -offset.value
                                isRescan = True
                            else:
                                editIssue += "end delete %d\n" % offset.value
                    elif clip.getClipType() == CLIPTYPE.SHOTNAME:
                        pass

                # not setup data
                self.excelMng.setRow(excelIndex, Column2.EDIT_ORDER.name, newEditOrder)
                self.excelMng.setRow(excelIndex, Column2.TC_IN.name, startTime)
                self.excelMng.setRow(excelIndex, Column2.TC_OUT.name, endTime)

                # Check Different TC Range
                if shotName:
                    pass
                elif findItem and findItem.has_key(Column2.SHOT_NAME.name.lower()):
                    shotName = findItem[Column2.SHOT_NAME.name.lower()]
                elif self.editTimeline.getShotName(editCutInFrameList[cutIndex]):
                    shotName = self.editTimeline.getShotName(editCutInFrameList[cutIndex])

                # if shotName and clip.getClipType() == CLIPTYPE.PREVIZ:
                #     shotName = ''

                if shotName:
                    for shotNameCheckIndex in range(1, self.excelMng.count() - 1):
                        alreadyShotName = self.excelMng.getRow(shotNameCheckIndex, Column2.SHOT_NAME.name)
                        alreadyClipName = self.excelMng.getRow(shotNameCheckIndex, Column2.CLIP_NAME.name)
                        if alreadyShotName == shotName and alreadyClipName == clipName and shotNameCheckIndex != excelIndex:
                            Msg.bold(shotName, clipName, excelIndex)
                            Msg.bold(alreadyShotName, alreadyClipName, shotNameCheckIndex)
                            alreadyTCIN = self.excelMng.getRow(shotNameCheckIndex, Column2.TC_IN.name)
                            alreadyTCOUT = self.excelMng.getRow(shotNameCheckIndex, Column2.TC_OUT.name)
                            alreadyFPS = self.excelMng.getRow(shotNameCheckIndex, Column2.SCAN_FPS.name)

                            try:
                                alreadyStartTC = otio.opentime.RationalTime.from_timecode(alreadyTCIN, float(alreadyFPS))
                                alreadyEndTC = otio.opentime.RationalTime.from_timecode(alreadyTCOUT, float(alreadyFPS))
                            except:
                                break
                            alreadyDurationTC = otio.opentime.RationalTime.duration_from_start_end_time(alreadyStartTC, alreadyEndTC)
                            alreadyTimeRange = otio.opentime.TimeRange(alreadyStartTC, alreadyDurationTC)

                            if findItem:
                                if findItem.has_key('mov_cut_fps'):
                                    itemFPS = findItem['mov_cut_fps']
                                else:
                                    itemFPS = findItem[Column2.SCAN_FPS.name.lower()]
                                itemStartTC = otio.opentime.RationalTime.from_timecode(findItem[Column2.TC_IN.name.lower()], float(itemFPS))
                                itemEndTC = otio.opentime.RationalTime.from_timecode(findItem[Column2.TC_OUT.name.lower()], float(itemFPS))
                                itemDurationTC = otio.opentime.RationalTime.duration_from_start_end_time(itemStartTC, itemEndTC)
                                itemTimeRange = otio.opentime.TimeRange(itemStartTC, itemDurationTC)

                                print itemTimeRange.start_time.to_timecode(), itemTimeRange.end_time_exclusive().to_timecode()
                                print alreadyTimeRange.start_time.to_timecode(), alreadyTimeRange.end_time_exclusive().to_timecode()
                                print clip.getTimeRange().start_time.to_timecode(), clip.getTimeRange().end_time_exclusive().to_timecode()

                                print itemTimeRange.contains(alreadyTimeRange)
                                print itemTimeRange.overlaps(alreadyTimeRange)
                                print itemTimeRange.contains(clip.getTimeRange())
                                print itemTimeRange.overlaps(clip.getTimeRange())

                                print clip.getTimeRange().contains(itemTimeRange)
                                print clip.getTimeRange().overlaps(itemTimeRange)
                                print clip.getTimeRange().contains(alreadyTimeRange)
                                print clip.getTimeRange().overlaps(alreadyTimeRange)

                                print alreadyTimeRange.contains(itemTimeRange)
                                print alreadyTimeRange.overlaps(itemTimeRange)
                                print alreadyTimeRange.contains(clip.getTimeRange())
                                print alreadyTimeRange.overlaps(clip.getTimeRange())


                            if (alreadyEndTC - alreadyStartTC).value > durationTime.value: # Before Long
                                editIssue = 'duplicate %s' % shotName
                                shotName = ''
                            else:
                                self.excelMng.setRow(shotNameCheckIndex, Column2.SHOT_NAME.name, '')
                                self.excelMng.setRow(shotNameCheckIndex, Column2.EDIT_ISSUE.name, 'duplicate %s' % alreadyShotName)
                            break

                self.excelMng.setRow(excelIndex, Column2.SHOT_NAME.name, shotName)

                if findItem and findItem.has_key(Column2.TYPE.name.lower()):
                    plateType = findItem[Column2.TYPE.name.lower()]

                if clip.getClipType() == CLIPTYPE.CLIP:
                    self.excelMng.setRow(excelIndex, Column2.TYPE.name, plateType)
                    plateIndex += 1

                if findItem and findItem.has_key(Column2.VERSION.name.lower()) and isRescan == False:
                    self.excelMng.setRow(excelIndex, Column2.VERSION.name, findItem[Column2.VERSION.name.lower()])
                else:
                    self.excelMng.setRow(excelIndex, Column2.VERSION.name, '')

                self.excelMng.setRow(excelIndex, Column2.FRAME_IN.name, frameIn)
                self.excelMng.setRow(excelIndex, Column2.FRAME_OUT.name, frameOut - 1)
                self.excelMng.setRow(excelIndex, Column2.CLIP_DURATION.name, shotDuration)
                self.excelMng.setRow(excelIndex, Column2.ORIGINAL_ROOT_FOLDER.name, "/stuff/%s/scan" % self.showName)

                if findItem and findItem.has_key(Column2.ORIGINAL_ROOT_PATH.name.lower() and isRescan == False):
                    self.excelMng.setRow(excelIndex, Column2.ORIGINAL_ROOT_PATH.name, findItem[Column2.ORIGINAL_ROOT_PATH.name.lower()])
                    self.excelMng.setRow(excelIndex, Column2.RESOLUTION.name, findItem[Column2.RESOLUTION.name.lower()])
                else:
                    self.excelMng.setRow(excelIndex, Column2.ORIGINAL_ROOT_PATH.name, '')
                    self.excelMng.setRow(excelIndex, Column2.RESOLUTION.name, '')

                self.excelMng.setRow(excelIndex, Column2.CLIP_NAME.name, clipName)
                self.excelMng.setRow(excelIndex, Column2.SCAN_DURATION.name, '')

                if clip.dissolve['end_offset'].value != 0 or clip.dissolve['top_offset'].value != 0:
                    Msg.bold("Dissolve 있음")
                    top_offset = clip.dissolve['top_offset']
                    end_offset = clip.dissolve['end_offset']
                    issue += "Dissolve|top_offset:%s|end_offset:%s\n" % (top_offset.value, end_offset.value)

                self.excelMng.setRow(excelIndex, Column2.ISSUE.name, issue.strip())
                self.excelMng.setRow(excelIndex, Column2.XML_NAME.name, self.editFilePath)
                self.excelMng.setRow(excelIndex, Column2.MOV_CUT_IN.name, movCutIn)

                # self.excelMng.setRow(excelIndex, Column2.MOV_CUT_DURATION.name, durationTime.value)
                self.excelMng.setRow(excelIndex, Column2.SCAN_FPS.name, "%0.2f" % clip.getFPS(ignore_ntsc=True))

                # Edit Check
                try:
                    # if editList[row - 1][Column2.EDIT_ORDER.name.lower()] != newEditOrder or editList[row - 1][Column2.CLIP_NAME.name.lower()] != clipName:
                    if editList[row - 1][Column2.CLIP_NAME.name.lower()] != clipName:
                        # Check List
                        # 1. Edit Order Change
                        # for oldIndex in range(row - 1, len(editList)):
                        #     if editList[oldIndex][Column2.CLIP_NAME.name.lower()] == clipName:
                        #         print clipName, editList[oldIndex][Column2.TC_IN.name.lower()], startTC
                        #         # tcIn <= inFrames.to_timecode() and tcOut >= endFrames.to_timecode()
                        #         # if editList[oldIndex][Column2.TC_IN.name.lower()]
                        #     # print editList[row - 1][Column2.EDIT_ORDER.name.lower()], editOrder
                        #     # print editList[row - 1][Column2.CLIP_NAME.name.lower()], clipName
                        # 2. New Shot
                        #   New Check Method
                        #   1) if not exists findItem
                        if not findItem and not shotName:
                            editIssue += "new shot\n"
                        # 3. Omit List
                        #   Omit Shot Check Method
                        #   1) before edit to not exists new clip
                        isOmit = True
                        editStartTime = otio.opentime.RationalTime.from_timecode(editList[row-1][Column2.TC_IN.name.lower()], fpsBase)
                        editEndTime = otio.opentime.RationalTime.from_timecode(editList[row - 1][Column2.TC_OUT.name.lower()], fpsBase)
                        editDurationTime = otio.opentime.RationalTime.duration_from_start_end_time(editStartTime, editEndTime)
                        editRange = otio.opentime.TimeRange(editStartTime, editDurationTime)

                        for newEditIndex in editCutInFrameList:
                            newClipData = self.editTimeline.getClipList(newEditIndex)
                            for newClipIndex, newClip in enumerate(newClipData):
                                # if newClip.clip.media_reference.is_missing_reference:  # source file missing
                                #     continue
                                newClipName = newClip.getClipName()

                                # if newClip.getTimeRange().contains(editRange) != newClip.getTimeRange().overlaps(editRange):
                                #     Msg.error(newClipName, newClip.getTimeRange(), editRange)
                                #     print newClip.getTimeRange().contains(editRange)
                                #     print newClip.getTimeRange().overlaps(editRange)
                                if editList[row - 1][Column2.CLIP_NAME.name.lower()] == newClipName and newClip.getTimeRange().overlaps(editRange):
                                    isOmit = False
                                    break

                            if isOmit == False:
                                break

                        if isOmit:
                            self.omitList.append((editList[row - 1][Column2.SHOT_NAME.name.lower()],
                                             editList[row - 1][Column2.CLIP_NAME.name.lower()],
                                                  editList[row - 1][Column2.TC_IN.name.lower()],
                                                  editList[row - 1][Column2.TC_OUT.name.lower()]))
                            # if findItem:
                            #     Msg.bold("OmitList ? [%s]" % editList[row - 1][Column2.CLIP_NAME.name.lower()])

                        # # 4. Re-Birth
                        # if findItem:
                        #     shotInfo = getShotInfo(self.showCode, findItem[Column2.SHOT_NAME.name.lower()])
                        #     if shotInfo and shotInfo[0]['status'] == 'Omit':
                        #         print "Re-Birth", findItem[Column2.SHOT_NAME.name.lower()]
                except Exception as e:
                    pass
                    # Msg.error(e.message, len(editList), row)
                self.excelMng.setRow(excelIndex, Column2.EDIT_ISSUE.name, editIssue.strip())

                rowIndex += 1

            cutIndex += 1
            editOrder += 1

        try:
            if self.editTimeline.getClipList(editCutInFrameList[-1])[0].isSlate():
                movCutIn = editCutInFrameList[-1]
                self.excelMng.setRow(rowIndex, Column2.MOV_CUT_IN.name, movCutIn)
            else:
                self.excelMng.setRow(rowIndex, Column2.MOV_CUT_IN.name, self.otioData.duration().value + self.globalStartFrame)
        except:
            self.excelMng.setRow(rowIndex, Column2.MOV_CUT_IN.name, self.otioData.duration().value + self.globalStartFrame)

    def save(self):
        if not os.path.exists(os.path.dirname(self.excelFilePath)):
            os.makedirs(os.path.dirname(self.excelFilePath))

        previewDir = os.path.join(self.dirpath, 'preview')
        if not os.path.exists(previewDir):
            os.makedirs(previewDir)

        # TRACTOR SETUP
        job = author.Job()
        job.title = '(EDITORIAL) %s' % os.path.basename(self.editFilePath)
        job.comment = 'sourcefile : ' + self.editFilePath
        job.service = TRACTOR.SERVICE_KEY
        job.maxactive = TRACTOR.MAX_ACTIVE
        job.tier = TRACTOR.TIER
        job.tags = [TRACTOR.TAGS]
        job.projects = [TRACTOR.PROJECT]
        job.priority = TRACTOR.PRIORITY

        # metadataList = [{"DEXTER":[]}] * (len(self.excelMng.excelList) - 1)
        topOffset = 0.0
        endOffset = 0.0
        keepMetaInfo = {}

        print self.globalStartFrame, self.globalEndFrame
        rollSize = 24 * 60 * 20
        rollNum = 1
        distributeRollFrame = self.globalStartFrame
        print "Roll Count :", (self.globalEndFrame - self.globalStartFrame) / rollSize

        inspectTimeline = None

        rootTask = author.Task(title='preview mov setup')
        notificationCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
        notificationCmd += ['BotMsg', '-a', '@%s' % getpass.getuser(), '-b', 'VelozBot', '-m',
                            FORMAT.PREVIEWOKMSG.format(PROJECT=self.showName, FILE=self.excelFilePath)]
        rootTask.addCommand(author.Command(argv=notificationCmd, service=TRACTOR.SERVICE_KEY))
        job.addChild(rootTask)

        editOrderList = []
        for index in range(1, self.excelMng.count() - 1):
            editOrder = int(self.excelMng.getRow(index, Column2.EDIT_ORDER.name))
            startTC = self.excelMng.getRow(index, Column2.TC_IN.name)
            endTC = self.excelMng.getRow(index, Column2.TC_OUT.name)
            clipName = self.excelMng.getRow(index, Column2.CLIP_NAME.name)
            movCutIn = int(self.excelMng.getRow(index, Column2.MOV_CUT_IN.name))

            if movCutIn >= distributeRollFrame:
                print movCutIn, distributeRollFrame
                print isinstance(inspectTimeline, otio.schema.Timeline)
                if isinstance(inspectTimeline, otio.schema.Timeline):
                    # SAVE
                    otio.adapters.write_to_file(inspectTimeline, self.inspectOtioFile.replace('.otio', '_R%d.otio' % rollNum))
                    rollNum += 1
                    del inspectTimeline
                    inspectTimeline = None

                # TRACK SETUP

                inspectTimeline = otio.schema.Timeline(name=self.filename)
                movTrack = otio.schema.Track(name='EditMOV Track')
                inspectTimeline.tracks.append(movTrack)
                distributeRollFrame += rollSize
                startEditOrder = editOrder
                rollTask = author.Task(title='R%d Preview' % rollNum)
                notificationCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
                notificationCmd += ['BotMsg', '-a', '@%s' % getpass.getuser(), '-b', 'VelozBot', '-m',
                                    FORMAT.ROLLOKMSG.format(PROJECT=self.showName, ROLENAME='R%d' % rollNum)]
                rollTask.addCommand(author.Command(argv=notificationCmd, service=TRACTOR.SERVICE_KEY))
                rootTask.addChild(rollTask)

            # Find Next Shot Cut In
            movCutDuration = int(self.excelMng.getRow(self.excelMng.count() - 1, Column2.MOV_CUT_IN.name)) - movCutIn

            for tmpIdx in range(index + 1, self.excelMng.count()):
                if not self.excelMng.getRow(tmpIdx, Column2.EDIT_ORDER.name):
                    break
                tmpEditOrder = int(self.excelMng.getRow(tmpIdx, Column2.EDIT_ORDER.name))
                if tmpEditOrder > editOrder:
                    movCutDuration = int(self.excelMng.getRow(tmpIdx, Column2.MOV_CUT_IN.name)) - movCutIn
                    break

            movFPS = float(self.excelMng.getRow(index, Column2.SCAN_FPS.name))
            issue = self.excelMng.getRow(index, Column2.ISSUE.name)
            # print editOrder, startTC, endTC, clipName, movCutIn, movFPS
            #
            if not editOrder in editOrderList:
                metadata = {'DEXTER':[]}
                if endOffset != 0.0:
                    metadata['DEXTER'].append({'START_TC': keepMetaInfo['START_TC'],
                                               'END_TC': keepMetaInfo['END_TC'],
                                               'CLIPNAME': keepMetaInfo['CLIPNAME'],
                                               'CUT_IN': movCutIn,
                                               'CUT_OUT': movCutIn + endOffset})
                    # print endOffset
                    endOffset = 0.0
                metadata['DEXTER'].append({'START_TC': startTC,
                                           'END_TC': endTC,
                                           'CLIPNAME': clipName})
                # print 'clipName, metadata:', clipName, metadata
                shotTask = author.Task(title='{EDITORDER} - {CLIPNAME}'.format(EDITORDER=editOrder, CLIPNAME=clipName))
                command = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
                command += ['rvio', '-v', self.movFilePath]
                command += ['-t', '%s-%s' % (movCutIn, movCutIn + movCutDuration - 1)]
                previewMovFileName = '{PREVIEWDIR}/{EDITORDER}_{FILENAME}.mov'.format(PREVIEWDIR=previewDir,
                                                                                   EDITORDER=editOrder,
                                                                                   FILENAME=clipName.replace(' ', '_'))
                # print 'previewMovFileName:', previewMovFileName
                previewTmpMovFileName = previewMovFileName.replace('.mov', '_tmp.mov')
                command += ['-o', previewTmpMovFileName]
                shotTask.addCommand(author.Command(argv=command, service=TRACTOR.SERVICE_KEY))

                TCCommand = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
                TCCommand += ['ffmpeg', '-y', '-i', previewTmpMovFileName, '-map', '0', '-map', '-0:d', '-c', 'copy']
                TCCommand += ['-timecode', otio.opentime.RationalTime(movCutIn, movFPS).to_timecode(), previewMovFileName]
                shotTask.addCommand(author.Command(argv=TCCommand, service=TRACTOR.SERVICE_KEY))

                rmCommand = ['rm', '-rf', previewTmpMovFileName]
                shotTask.addCommand(author.Command(argv=rmCommand, service=TRACTOR.SERVICE_KEY))

                rollTask.addChild(shotTask)

                mediaReference = otio.schema.ExternalReference(available_range=otio.opentime.TimeRange(otio.opentime.RationalTime(movCutIn, movFPS),
                                                                                otio.opentime.RationalTime(movCutDuration, movFPS)),
                                                               target_url=previewMovFileName)
                movClip = otio.schema.Clip(name=clipName,
                                           source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(movCutIn, movFPS),
                                                                                otio.opentime.RationalTime(movCutDuration, movFPS)),
                                           media_reference=mediaReference,
                                           metadata=metadata)
                movTrack.append(movClip)
                editOrderList.append(editOrder)
            else:
                # print "!!!!"
                metadata = movTrack[-1].metadata
                metadata['DEXTER'].append({'START_TC': startTC,
                                           'END_TC': endTC,
                                           'CLIPNAME': clipName})
                movTrack[-1].metadata.update(metadata)

            if 'Dissolve' in issue:
                splitIssueList = issue.split('\n')
                for splitIssue in splitIssueList:
                    if "Dissolve" in splitIssue:
                        topOffset = float(splitIssue.split('|')[-2].split(':')[-1])
                        endOffset = float(splitIssue.split('|')[-1].split(':')[-1])
                        keepMetaInfo = {'START_TC': startTC,
                                        'END_TC': endTC,
                                        'CLIPNAME': clipName}
                        if topOffset != 0.0:
                            try:
                                metadata = movTrack[-2].metadata
                                endFrame = movTrack[-2].source_range.end_time_exclusive().to_frames()
                                metadata['DEXTER'].append({'START_TC': keepMetaInfo['START_TC'],
                                                           'END_TC': keepMetaInfo['END_TC'],
                                                           'CLIPNAME': keepMetaInfo['CLIPNAME'],
                                                           'CUT_IN': endFrame - topOffset,
                                                           'CUT_OUT': endFrame})
                                movTrack[-2].metadata.update(metadata)
                            except:
                                print '### ERROR ### movTrack:', movTrack
                            topOffset = 0.0
                        break

        if inspectTimeline != None:
            if rollNum != 1:
                otio.adapters.write_to_file(inspectTimeline, self.inspectOtioFile.replace('.otio', '_R%d.otio' % rollNum))
            else:
                otio.adapters.write_to_file(inspectTimeline, self.inspectOtioFile)

            author.setEngineClientParam(hostname=TRACTOR.IP, port=TRACTOR.PORT, user=getpass.getuser(), debug=True)
            job.spool()
            author.closeEngineClient()

        rowIndex = self.excelMng.count('omit_list')
        for omitData in self.omitList:
            self.excelMng.setRow(rowIndex, Column2.EDIT_ORDER.name, 'OmitList', sheet='omit_list')
            self.excelMng.setRow(rowIndex, Column2.SHOT_NAME.name, omitData[0], sheet='omit_list')
            self.excelMng.setRow(rowIndex, Column2.CLIP_NAME.name, omitData[1], sheet='omit_list')
            self.excelMng.setRow(rowIndex, Column2.TC_IN.name, omitData[2], sheet='omit_list')
            self.excelMng.setRow(rowIndex, Column2.TC_OUT.name, omitData[3], sheet='omit_list')
            rowIndex += 1

        self.excelMng.save(self.excelFilePath)
