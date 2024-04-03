#coding:utf-8
import opentimelineio as otio
import os
import xlwt
from Define import Column
import Msg
from calculator import *

class FCP7XMLParser():
    def __init__(self, xmlFilePath, showName=''):
        self.editFilePath = xmlFilePath
        filename, extension = os.path.splitext(self.editFilePath)
        self.excelFilePath = filename + '.xls'
        self.inspectOtioFile = filename + '.otio'

        self.showName = showName
        self.showName = self.getShowName()
        assert self.showName != '', "Not Found Show Name"

        self.openFileData()
        self.globalStartFrame = self.getGlobalStartFrame()
        self.cleanupTimeline = cleanupTrackData(self.otioData)

    def getShowName(self):
        if self.showName:
            return self.showName
        if self.editFilePath.startswith('/prod_nas'):
            showName = self.editFilePath.split('/')[3].lower()
            return showName
        return ''

    def openFileData(self):
        '''
        write xls file & read xml file
        :return:
        '''
        self.excelData = xlwt.Workbook(encoding='utf-8')
        self.excelSheet = self.excelData.add_sheet('scan_list')
        for column in Column:
            self.excelSheet.write(0, column.value, column.name.lower())
        self.otioData = otio.adapters.read_from_file(self.editFilePath, 'fcp_xml')
        self.inspectTimeline = otio.schema.Timeline()

    def getGlobalStartFrame(self):
        try:
            return self.otioData.global_start_time.to_frames()
        except:
            return 0

    def doIt(self):
        editCutInFrameList = sorted(self.cleanupTimeline.keys())
        cutIndex = 0
        rowIndex = 1
        editOrder = 1

        # Base Track - 0 MOV Clip
        # movTrack = otio.schema.Track(name='EditMOV Track')
        # print self.otioData.global_start_time, self.otioData.duration
        # movClip = otio.schema.Clip(name='editMov', source_range=self.otioData)

        self.inspectTimeline.video_tracks()
        while cutIndex < len(editCutInFrameList):
            clipData = self.cleanupTimeline[editCutInFrameList[cutIndex]]['clip']
            hasAfterSpeedRamp = False  # if True, row Index += 1
            for clipIndex, clip in enumerate(clipData):
                row = rowIndex
                issue = ''

                if clip.media_reference.is_missing_reference:  # source file missing
                    editOrder -= 1
                    break

                # Check Time Base
                if clip.media_reference.available_range is None:
                    continue

                inValue = (clip.source_range.start_time - clip.available_range().start_time).to_frames()

                outValue = (clip.source_range.end_time_exclusive() - clip.available_range().start_time).to_frames()

                fpsBase = clip.visible_range().start_time.rate
                startTime = clip.visible_range().start_time
                durationTime = clip.visible_range().duration

                # Disable NTSC
                if clip.media_reference.metadata.get('fcp_xml'):
                    fpsBase = float(clip.media_reference.metadata['fcp_xml']['rate']['timebase'])
                    startTime = clip.visible_range().start_time.rescaled_to(fpsBase)
                    durationTime = otio.opentime.RationalTime(durationTime.to_frames(), fpsBase)

                orgDurationTime = durationTime
                editTCRange = otio.opentime.TimeRange(startTime, durationTime)

                # addTcIn = otio.opentime.RationalTime(0, fpsBase)
                # addTcOut = otio.opentime.RationalTime(0, fpsBase)
                addMovCutDuration = otio.opentime.RationalTime(0, fpsBase)

                # Write Data
                self.excelSheet.write(row, Column.EDIT_ORDER.value, editOrder)

                # SpeedRamp Variables
                isSpeedRamp = None
                retimeSpeed = '0'
                postOrderRetimeSpeed = '0'

                updateInValue = inValue
                updateOutValue = outValue

                # Check Effects (Time Remap[Retime Value, InValue OutValue], TODO:Basic Motion[Anchor Offset, Rotation, Scales]
                if hasattr(clip, 'effects'):
                    retimeSpeed = getRetime(clip)

                    if retimeSpeed != '0' and float(retimeSpeed) >= 10:  # has Retime
                        updateInValue = getKeyframeValue(clip, updateInValue)  # Retime TC Interpolation
                        updateOutValue = getKeyframeValue(clip, updateOutValue)  # Retime TC Interpolation
                        if updateInValue != inValue:
                            startTime = calcRetimeTC(clip, int(round(updateInValue)), fpsBase)
                        if updateOutValue != outValue:
                            # updateOutValue = updateOutValue - offset
                            updateOutValue = updateOutValue - (float(retimeSpeed) * 0.01)
                            if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                                out_offset = self.cleanupTimeline[editCutInFrameList[cutIndex]]['out_offset']
                                updateOutValue = updateOutValue + ((float(retimeSpeed) * 0.01) * out_offset.to_frames())

                            updateOutValue = roundIOValue(updateOutValue, retimeSpeed)
                            endTime = calcRetimeTC(clip, updateOutValue, fpsBase)

                            durationTime = endTime - startTime

                        try:
                            preOrderClip = self.cleanupTimeline[editCutInFrameList[cutIndex - 1]]['clip'][clipIndex]
                            preOrderReelName = os.path.basename(preOrderClip.media_reference.target_url).split('.')[0]
                        except Exception as e:
                            preOrderClip = None
                            preOrderReelName = ''
                            Msg.error(e.message)
                        try:
                            postOrderClip = self.cleanupTimeline[editCutInFrameList[cutIndex + 1]]['clip'][clipIndex]
                            postOrderReelName = os.path.basename(postOrderClip.media_reference.target_url).split('.')[0]
                        except Exception as e:
                            postOrderClip = None
                            postOrderReelName = ''
                            Msg.error(e.message)

                        curReelName = os.path.basename(clip.media_reference.target_url).split('.')[0]

                        if preOrderReelName == curReelName:
                            Msg.warning("PreOrder Clip Same", preOrderClip.name, clip.name)
                            isSpeedRamp = "Before"

                            preOrderStartTime = preOrderClip.visible_range().start_time.rescaled_to(fpsBase)
                            preOrderDurationTime = otio.opentime.RationalTime(
                                preOrderClip.visible_range().duration.to_frames(), fpsBase)

                            # addTcIn += preOrderDurationTime
                            # addTcOut += preOrderDurationTime
                            addMovCutDuration += preOrderClip.trimmed_range().duration

                            if hasattr(preOrderClip, "effects"):
                                Msg.warning('Before Effect!!')
                                preOrderSpeedTime = getRetime(postOrderClip)
                                # preOrderReverse = getRetimeReverse(preOrderClip)
                                print preOrderSpeedTime
                                for effect in preOrderClip.effects:
                                    print effect

                            editTCRange = otio.opentime.TimeRange(preOrderStartTime,
                                                                  editTCRange.end_time_exclusive() - preOrderStartTime)

                            # Write Excel
                            speedRampText = "SpeedRamp {RETIME}%\n".format(RETIME=retimeSpeed)
                            retimeRange = otio.opentime.TimeRange(startTime, durationTime)

                            issue += "SpeedRamp TCInfo: %s - %s\n" % (retimeRange.start_time.to_timecode(),
                                                                      (
                                                                                  retimeRange.end_time_exclusive() + otio.opentime.RationalTime(
                                                                              1 * (int(float(retimeSpeed)) * 0.01),
                                                                              fpsBase)).to_timecode())

                            self.excelSheet.write(row, Column.RETIME.value, speedRampText)

                        elif postOrderReelName == curReelName:
                            Msg.warning("postOrder Clip Same")
                            isSpeedRamp = "After"
                            hasAfterSpeedRamp = True
                            afterHasEffect = False

                            postOrderRetimeSpeed = "0"
                            postOrderStartTime = postOrderClip.visible_range().start_time.rescaled_to(fpsBase)
                            postOrderDurationTime = otio.opentime.RationalTime(
                                postOrderClip.visible_range().duration.to_frames() - 1, fpsBase)
                            postOrderEndTime = postOrderStartTime + postOrderDurationTime

                            if hasattr(postOrderClip, "effects"):
                                Msg.warning("After Effect!!")

                                postOrderRetimeSpeed = getRetime(postOrderClip)
                                # postOrderReverse = getRetimeReverse(postOrderClip)
                                if postOrderRetimeSpeed != "0" and float(postOrderRetimeSpeed) >= 10:
                                    afterHasEffect = True
                                    postOrderInValue = getInValue(postOrderClip)
                                    postOrderOutValue = getOutValue(postOrderClip)

                                    postOrderInValue = getKeyframeValue(postOrderClip, postOrderInValue)
                                    postOrderOutValue = getKeyframeValue(postOrderClip, postOrderOutValue)

                                    postOrderStartTime = calcRetimeTC(postOrderClip, int(round(postOrderInValue)),
                                                                      fpsBase)
                                    # end OutValue calculation
                                    postOrderOutValue = postOrderOutValue - (int(postOrderRetimeSpeed) * 0.01)

                                    Msg.bold(postOrderOutValue)
                                    postOrderOutValue = roundIOValue(postOrderOutValue, postOrderRetimeSpeed)
                                    Msg.bold(postOrderOutValue)

                                    postOrderEndTime = calcRetimeTC(postOrderClip, postOrderOutValue, fpsBase)
                                    postOrderDurationTime = otio.opentime.RationalTime.duration_from_start_end_time(
                                        postOrderStartTime, postOrderEndTime)

                            addMovCutDuration += postOrderClip.trimmed_range().duration

                            clipDuration = postOrderEndTime - startTime

                            editTCRange = otio.opentime.TimeRange(startTime, clipDuration)

                            # Write Excel
                            speedRampText = "SpeedRamp {RETIME}%\n".format(RETIME=retimeSpeed)
                            retimeRange = otio.opentime.TimeRange(startTime, durationTime)

                            issue += "SpeedRamp TCInfo: %s - %s\n" % (retimeRange.start_time.to_timecode(),
                                                                      (
                                                                                  retimeRange.end_time_exclusive() + otio.opentime.RationalTime(
                                                                              1 * (int(postOrderRetimeSpeed) * 0.01),
                                                                              fpsBase)).to_timecode())

                            if afterHasEffect:
                                speedRampText += "SpeedRamp {RETIME}%".format(RETIME=postOrderRetimeSpeed)
                                retimeRange = otio.opentime.TimeRange(postOrderStartTime, postOrderDurationTime)
                                issue += "SpeedRamp TCInfo: %s - %s\n" % (retimeRange.start_time.to_timecode(),
                                                                          (
                                                                              retimeRange.end_time_exclusive()).to_timecode())

                            self.excelSheet.write(row, Column.RETIME.value, speedRampText)

                        else:  # Single Retime
                            editTCRange = otio.opentime.TimeRange(startTime, durationTime)

                            self.excelSheet.write(row, Column.RETIME.value,
                                                  "Retime {RETIME}%".format(RETIME=retimeSpeed))
                            issue += "Retime {RETIME}%\n".format(RETIME=retimeSpeed)

                plateType = 'main1'

                if clipIndex != 0:
                    plateType = 'src%d' % clipIndex

                if (isSpeedRamp or (retimeSpeed != '0' and float(retimeSpeed) >= 10)):
                    plateType += "_org"

                if len(clipData) > 1:
                    issue += 'has source\n'
                    Msg.warning('has source!!')

                self.excelSheet.write(row, Column.TYPE.value, plateType)
                # sheet.write(row, Column.VERSION.value, "v001")
                self.excelSheet.write(row, Column.FRAME_IN.value, 1001)
                self.excelSheet.write(row, Column.FRAME_OUT.value,
                                      1001 + orgDurationTime.to_frames() + addMovCutDuration.to_frames() - 1)
                self.excelSheet.write(row, Column.SHOT_DURATION.value,
                                      orgDurationTime.to_frames() + addMovCutDuration.to_frames())
                # sheet.write(row, Column.SCALE.value, "")
                self.excelSheet.write(row, Column.ORIGINAL_ROOT_FOLDER.value, "/stuff/%s/scan" % self.showName)
                # self.excelSheet.write(row, Column.ORIGINAL_ROOT_PATH.value, "")
                if "prv" not in os.path.basename(clip.media_reference.target_url).split('.')[0]:
                    self.excelSheet.write(row, Column.EXR_FILENAME.value, os.path.basename(clip.media_reference.target_url).split('.')[0])

                self.excelSheet.write(row, Column.TC_IN.value, editTCRange.start_time.to_timecode())
                if (retimeSpeed != "0" and float(retimeSpeed) >= 10) or (postOrderRetimeSpeed == '0' and float(postOrderRetimeSpeed) >= 10):
                    self.excelSheet.write(row, Column.TC_OUT.value, editTCRange.end_time_exclusive().to_timecode())
                else:
                    self.excelSheet.write(row, Column.TC_OUT.value, editTCRange.end_time_inclusive().to_timecode())

                self.excelSheet.write(row, Column.SCAN_DURATION.value, "%s" % editTCRange.duration.to_frames())

                if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('in_offset') or \
                        self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                    issue += "Dissolve 있음\n"
                    Msg.warning("Dissolve 있음")

                self.excelSheet.write(row, Column.ISSUE.value, issue.strip())
                # sheet.write(row, Column.RESOLUTION.value, "1920x1080")

                self.excelSheet.write(row, Column.XML_NAME.value, self.editFilePath)
                self.excelSheet.write(row, Column.MOV_CUT_IN.value,
                                      self.globalStartFrame + editCutInFrameList[cutIndex])
                try:
                    self.excelSheet.write(row, Column.MOV_CUT_DURATION.value,
                                          editCutInFrameList[cutIndex + 1] - editCutInFrameList[
                                              cutIndex] + addMovCutDuration.to_frames())
                except Exception as e:
                    self.excelSheet.write(row, Column.MOV_CUT_DURATION.value, clip.trimmed_range().duration.to_frames())
                    Msg.error(e.message)

                ntscFps = fpsBase * (1000.0 / 1001)
                self.excelSheet.write(row, Column.SCAN_FPS.value, "%0.2f" % ntscFps)

                self.excelSheet.row(row).height_mismatch = True
                self.excelSheet.row(row).height = 256 + (256 * (issue.strip().count('\n')))

                rowIndex += 1

            if hasAfterSpeedRamp:
                cutIndex += 1

            cutIndex += 1
            editOrder += 1

    def save(self):
        if not os.path.exists(os.path.dirname(self.excelFilePath)):
            os.makedirs(os.path.dirname(self.excelFilePath))

        self.excelData.save(self.excelFilePath)