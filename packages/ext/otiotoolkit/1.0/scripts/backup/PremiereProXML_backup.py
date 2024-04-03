#coding:utf-8
from core import listMessageBox
import xlwt
from Define import *
import Msg
from calculator import *
import DBConfig

class PremiereProXMLParser():
    def __init__(self, xmlFilePath, showName='', plateType='main'):
        self.editFilePath = xmlFilePath
        filename, extension = os.path.splitext(self.editFilePath)
        self.filename = filename
        self.excelFilePath = filename + '.xls'
        self.inspectOtioFile = filename + '.otio'
        self.movFilePath = filename + '.mov'
        self.plateType = plateType

        self.showName = showName
        self.showName = self.getShowName()
        assert self.showName != '', "Not Found Show Name"
        self.showCode = getShowCode(self.showName)

        self.coll = DBConfig.db[self.showName]

        # Check Edit Changed
        editFileList = DBConfig.getEditFileList(self.coll)
        self.diffEditFile = ""
        self.toVelozStatus = ""
        if editFileList:
            box = listMessageBox.QListMessageBox(reversed(editFileList))
            box.listWidget.setCurrentRow(0)
            box.exec_()
            if box.result == True:
                self.diffEditFile = box.listWidget.selectedItems()[0].text()
                self.toVelozStatus = box.velozStatus

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
        for column in Column2:
            self.excelSheet.write(0, column.value, column.name.lower())
        self.otioData = otio.adapters.read_from_file(self.editFilePath, 'fcp_xml')
        self.inspectTimeline = otio.schema.Timeline(name=os.path.basename(self.filename))

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
        movTrack = otio.schema.Track(name='EditMOV Track')
        self.inspectTimeline.tracks.append(movTrack)
        mediaReference = otio.schema.ExternalReference(
            available_range=otio.opentime.TimeRange(self.otioData.global_start_time, self.otioData.duration()),
            target_url=self.movFilePath)

        editList = []
        if self.diffEditFile:
            editList = DBConfig.editList(self.coll, self.diffEditFile)

        omitList = []

        while cutIndex < len(editCutInFrameList):
            clipData = self.cleanupTimeline[editCutInFrameList[cutIndex]]['clip']
            hasAfterSpeedRamp = False  # if True, row Index += 1
            mainClipName = ''
            metadataList = []
            mainFpsBase = 24
            mainDuration = 0
            for clipIndex, clip in enumerate(clipData):
                row = rowIndex
                issue = ''

                if clip.media_reference.is_missing_reference:  # source file missing
                    editOrder -= 1
                    break

                # Check Time Base
                if clip.media_reference.available_range is None:
                    continue

                fpsBase = clip.visible_range().start_time.rate
                startTime = clip.visible_range().start_time
                durationTime = clip.visible_range().duration

                # Disable NTSC
                if clip.media_reference.metadata.get('fcp_xml') and clip.media_reference.metadata['fcp_xml']['rate'].get('ntsc') == "TRUE":
                    fpsBase = float(clip.media_reference.metadata['fcp_xml']['rate']['timebase'])
                    startTime = clip.visible_range().start_time.rescaled_to(fpsBase)
                    durationTime = otio.opentime.RationalTime(durationTime.to_frames(), fpsBase)

                orgDurationTime = durationTime
                editTCRange = otio.opentime.TimeRange(startTime, durationTime)

                addMovCutDuration = otio.opentime.RationalTime(0, fpsBase)

                # SpeedRamp Variables
                isSpeedRamp = None
                retimeSpeed = '0'
                postOrderRetimeSpeed = '0'

                # Check Effects (Time Remap[Retime Value, InValue OutValue], TODO:Basic Motion[Anchor Offset, Rotation, Scales]
                if hasattr(clip, 'effects'):
                    retimeSpeed, isReverse = getRetime(clip)
                    scale, rotation, center, anchor = getTransformValue(clip)
                    # fRetimeSpeed = float(retimeSpeed)
                    iRetimeSpeed = int(float(retimeSpeed))

                    if retimeSpeed != '0' and availableRetimeSpeed(retimeSpeed):  # has Retime
                        retimeInValue = getKeyframeValue2(clip, STRING.RETIME_IN)  # Retime TC Interpolation
                        retimeOutValue = getKeyframeValue2(clip, STRING.RETIME_OUT)  # Retime TC Interpolation

                        if retimeInValue is None:
                            # print startTime.to_timecode(), os.path.basename(clip.media_reference.target_url).split('.')[0]
                            retimeInValue = startTime.value

                        if retimeOutValue is None:
                            retimeOutValue = editTCRange.end_time_exclusive().value

                        # startTime = calcRetimeTC(clip, int(round(updateInValue)), fpsBase)
                        # Calc Start Retime
                        if isReverse:
                            startValue = clip.available_range().start_time.value + (retimeInValue * (-iRetimeSpeed * 0.01))
                        else:
                            startValue = clip.available_range().start_time.value + retimeInValue
                        startTime = otio.opentime.RationalTime(startValue, fpsBase)

                        # Calc End Retime
                        if isReverse:
                            endValue = startTime.value - ((durationTime.value - 1) * (-iRetimeSpeed * 0.01))
                        else:
                            retimeOutValue = retimeOutValue - (iRetimeSpeed * 0.01)
                            if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                                out_offset = self.cleanupTimeline[editCutInFrameList[cutIndex]]['out_offset']
                                retimeOutValue = retimeOutValue + ((iRetimeSpeed * 0.01) * out_offset.to_frames())

                            retimeOutValue = roundIOValue(retimeOutValue, retimeSpeed)
                            endValue = clip.available_range().start_time.value + retimeOutValue

                        endTime = otio.opentime.RationalTime(endValue, fpsBase)
                        durationTime = otio.opentime.RationalTime.duration_from_start_end_time(startTime, endTime)

                        try:
                            preOrderClip = self.cleanupTimeline[editCutInFrameList[cutIndex - 1]]['clip'][clipIndex]
                            preOrderReelName = os.path.basename(preOrderClip.media_reference.target_url).split('.')[0]
                            preOrderRetimeSpeed, isPreOrderReverse = getRetime(preOrderClip)
                        except Exception as e:
                            preOrderClip = None
                            preOrderReelName = ''
                            preOrderRetimeSpeed = "0"
                            isPreOrderReverse = False
                            # Msg.error(e.message)
                        try:
                            postOrderClip = self.cleanupTimeline[editCutInFrameList[cutIndex + 1]]['clip'][clipIndex]
                            postOrderReelName = os.path.basename(postOrderClip.media_reference.target_url).split('.')[0]
                            postOrderRetimeSpeed, isPostOrderReverse = getRetime(postOrderClip)
                        except Exception as e:
                            postOrderClip = None
                            postOrderReelName = ''
                            postOrderRetimeSpeed = "0"
                            isPostOrderReverse = False
                            # Msg.error(e.message)

                        curReelName = os.path.basename(clip.media_reference.target_url).split('.')[0]

                        if preOrderReelName == curReelName and isReverse == isPreOrderReverse:
                            Msg.warning("PreOrder Clip Same", preOrderClip.name, clip.name)
                            isSpeedRamp = "Before"
                            preOrderHasEffect = False

                            # preOrderRetimeSpeed = "0"
                            preOrderStartTime = preOrderClip.visible_range().start_time.rescaled_to(fpsBase)
                            preOrderDurationTime = otio.opentime.RationalTime(preOrderClip.visible_range().duration.value - 1, fpsBase)
                            preOrderEndTime = preOrderStartTime + preOrderDurationTime

                            if hasattr(preOrderClip, "effects"):
                                Msg.warning('Before Effect!!')

                                preOrderScale, preOrderRotation, preOrderCenter, preOrderAnchor = getTransformValue(preOrderClip)
                                iPreOrderRetimeSpeed = int(float(preOrderRetimeSpeed))

                                if preOrderRetimeSpeed != "0" and availableRetimeSpeed(postOrderRetimeSpeed):
                                    preOrderHasEffect = True
                                    preOrderRetimeInValue = getKeyframeValue2(preOrderClip, STRING.RETIME_IN)  # Retime TC Interpolation
                                    preOrderRetimeOutValue = getKeyframeValue2(preOrderClip, STRING.RETIME_OUT)  # Retime TC Interpolation

                                    # Calc Start Retime
                                    if isPreOrderReverse:
                                        preOrderStartValue = preOrderClip.available_range().start_time.value + ( preOrderRetimeInValue * (-iPreOrderRetimeSpeed * 0.01))
                                    else:
                                        preOrderStartValue = preOrderClip.available_range().start_time.value + preOrderRetimeInValue
                                    preOrderStartTime = otio.opentime.RationalTime(preOrderStartValue, fpsBase)

                                    # Calc End Retime
                                    if isPreOrderReverse:
                                        preOrderEndValue = preOrderStartTime.value - ((preOrderDurationTime.value - 1) * (-iPreOrderRetimeSpeed * 0.01))
                                    else:
                                        preOrderRetimeOutValue = preOrderRetimeOutValue - (iPreOrderRetimeSpeed * 0.01)
                                        if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                                            out_offset = self.cleanupTimeline[editCutInFrameList[cutIndex]]['out_offset']
                                            preOrderRetimeOutValue = preOrderRetimeOutValue + ((iPreOrderRetimeSpeed * 0.01) * out_offset.to_frames())

                                        preOrderRetimeOutValue = roundIOValue(preOrderRetimeOutValue, preOrderRetimeSpeed)
                                        preOrderEndValue = preOrderClip.avaliable_range().start_time.value + preOrderRetimeOutValue

                                    preOrderEndTime = otio.opentime.RationalTime(preOrderEndValue, fpsBase)
                                    preOrderDurationTime = otio.opentime.RationalTime.duration_from_start_end_time(preOrderStartTime, preOrderEndTime)


                            addMovCutDuration += preOrderClip.trimmed_range().duration

                            editTCRange = otio.opentime.TimeRange(preOrderStartTime, editTCRange.end_time_exclusive() - preOrderStartTime)

                            # Write Excel
                            speedRampText = ""
                            issue = ""
                            if preOrderHasEffect:
                                # speedRampText += FORMAT.SPEEDRAMP.format()
                                preOrderRetimeRange = otio.opentime.TimeRange(preOrderStartTime, preOrderDurationTime)
                                issue += FORMAT.SPEEDRAMPTC.format(RETIME=preOrderRetimeSpeed,
                                                                   TCIN=preOrderRetimeRange.start_time.to_timecode(),
                                                                   TCOUT=preOrderRetimeRange.end_time_exclusive().to_timecode())

                            # speedRampText += FORMAT.SPEEDRAMP.format()
                            retimeRange = otio.opentime.TimeRange(startTime, durationTime)
                            issue += FORMAT.SPEEDRAMPTC.format(RETIME=retimeSpeed,
                                                               TCIN=retimeRange.start_time.to_timecode(),
                                                               TCOUT=(retimeRange.end_time_exclusive() + otio.opentime.RationalTime(1 * (iRetimeSpeed * 0.01), fpsBase)).to_timecode())
                        elif postOrderReelName == curReelName and isReverse == isPostOrderReverse:
                            Msg.warning("postOrder Clip Same")
                            isSpeedRamp = "After"
                            postOrderHasEffect = False

                            # postOrderRetimeSpeed = "0"
                            postOrderStartTime = postOrderClip.visible_range().start_time.rescaled_to(fpsBase)
                            postOrderDurationTime = otio.opentime.RationalTime(postOrderClip.visible_range().duration.value - 1, fpsBase)
                            postOrderEndTime = postOrderStartTime + postOrderDurationTime

                            if hasattr(postOrderClip, "effects"):
                                Msg.warning("After Effect!!")
                                postOrderScale, postOrderRotation, postOrderCenter, postOrderAnchor = getTransformValue(postOrderClip)
                                iPostOrderRetimeSpeed = int(float(postOrderRetimeSpeed))

                                if postOrderRetimeSpeed != "0" and availableRetimeSpeed(postOrderRetimeSpeed):
                                    afterHasEffect = True
                                    hasAfterSpeedRamp = True
                                    postOrderRetimeInValue = getKeyframeValue2(postOrderClip, STRING.RETIME_IN)
                                    postOrderRetimeOutValue = getKeyframeValue2(postOrderClip, STRING.RETIME_IN)

                                    # Calc Start Retime
                                    if isPostOrderReverse:
                                        postOrderStartValue = postOrderClip.available_range().start_time.value + (postOrderRetimeInValue * (-iPostOrderRetimeSpeed * 0.01))
                                    else:
                                        postOrderStartValue = postOrderClip.available_range().start_time.value + postOrderRetimeInValue
                                    postOrderStartTime = otio.opentime.RationalTime(postOrderStartValue, fpsBase)

                                    # Calc End Retime
                                    if isPostOrderReverse:
                                        postOrderEndValue = postOrderStartTime.value - ((postOrderDurationTime.value - 1) * (-iPostOrderRetimeSpeed * 0.01))
                                    else:
                                        postOrderRetimeOutValue = postOrderRetimeOutValue - (iPostOrderRetimeSpeed * 0.01)
                                        if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                                            out_offset = self.cleanupTimeline[editCutInFrameList[cutIndex]]['out_offset']
                                            postOrderRetimeOutValue = postOrderRetimeOutValue + ((iPostOrderRetimeSpeed * 0.01) * out_offset.to_frames())

                                        postOrderRetimeOutValue = roundIOValue(postOrderRetimeOutValue, postOrderRetimeSpeed)
                                        postOrderEndValue = postOrderClip.available_range().start_time.value + postOrderRetimeOutValue

                                    postOrderEndTime = otio.opentime.RationalTime(postOrderEndValue, fpsBase)
                                    postOrderDurationTime = otio.opentime.RationalTime.duration_from_start_end_time(postOrderStartTime, postOrderEndTime)

                            addMovCutDuration += postOrderClip.trimmed_range().duration

                            clipDuration = postOrderEndTime - startTime
                            editTCRange = otio.opentime.TimeRange(startTime, clipDuration)

                            # Write Excel
                            # speedRampText = FORMAT.SPEEDRAMP.format()
                            retimeRange = otio.opentime.TimeRange(startTime, durationTime)
                            issue += FORMAT.SPEEDRAMPTC.format(RETIME=retimeSpeed,
                                                               TCIN=retimeRange.start_time.to_timecode(),
                                                               TCOUT=(retimeRange.end_time_exclusive() + otio.opentime.RationalTime(1 * (int(float(postOrderRetimeSpeed)) * 0.01), fpsBase)).to_timecode())

                            if postOrderHasEffect:
                                # speedRampText += FORMAT.SPEEDRAMP.format()
                                postOrderRetimeRange = otio.opentime.TimeRange(postOrderStartTime, postOrderDurationTime)
                                issue += FORMAT.SPEEDRAMPTC.format(RETIME=postOrderRetimeSpeed,
                                                                   TCIN=postOrderRetimeRange.start_time.to_timecode(),
                                                                   TCOUT=(postOrderRetimeRange.end_time_exclusive()).to_timecode())

                        else:  # Single Retime
                            editTCRange = otio.opentime.TimeRange(startTime, durationTime)
                            issue += FORMAT.RETIME.format(RETIME=retimeSpeed)

                    if scale != DEFAULT.SCALE:
                        issue += FORMAT.SCALE.format(SCALE=scale)
                    if rotation != DEFAULT.ROTATION:
                        issue += FORMAT.ROTATION.format(ROTATE=rotation)
                    # if center != DEFAULT.CENTER:
                    #     issue += FORMAT.CENTER.format(CENTER=center)
                    # if anchor != DEFAULT.ANCHOR_POINT:
                    #     issue += FORMAT.ANCHOR_POINT.format(ANCHOR=anchor)

                # plateType = 'main1'
                #
                # if clipIndex != 0:
                #     plateType = 'src%d' % clipIndex

                plateType = '%s%d' % (self.plateType, clipIndex + 1)

                if (isSpeedRamp or (retimeSpeed != '0' and availableRetimeSpeed(retimeSpeed))):
                    plateType += "_org"

                if len(clipData) > 1:
                    issue += 'has source\n'
                    Msg.warning('has source!!')

                # Value Setup
                if (retimeSpeed != "0" and availableRetimeSpeed(retimeSpeed)) or (
                        postOrderRetimeSpeed == '0' and availableRetimeSpeed(postOrderRetimeSpeed)):
                    endTC = editTCRange.end_time_exclusive()
                else:
                    # endTC = editTCRange.end_time_inclusive()
                    endTC = editTCRange.end_time_exclusive() - otio.opentime.RationalTime(1, fpsBase)

                clipName = os.path.basename(clip.media_reference.target_url).split('.')[0]
                startTC = editTCRange.start_time.to_timecode()
                frameIn = 1001
                frameOut = 1001 + orgDurationTime.to_frames() + addMovCutDuration.to_frames() - 1

                # Get DB Info
                findItem = DBConfig.getData(self.coll, '', clipName, startTC, endTC.to_timecode())

                if findItem:
                    # Check Different TC Range
                    if startTC != findItem[Column2.TC_IN.name.lower()]:
                        # Msg.warning("startTC :", startTC, "alreadyTC_IN :", findItem[Column2.TC_IN.name.lower()])
                        originalTC_IN = otio.opentime.RationalTime.from_timecode(findItem[Column2.TC_IN.name.lower()], fpsBase)
                        offset = originalTC_IN - editTCRange.start_time
                        frameIn = findItem[Column2.FRAME_IN.name.lower()]
                        frameIn -= offset.value
                        if offset.value <= 0: # nag value
                            issue += "top delete %d\n" % -offset.value
                        else:
                            issue += "top add %d\n" % offset.value
                    if endTC.to_timecode() != findItem[Column2.TC_OUT.name.lower()]:
                        # Msg.warning("endTC :", endTC.to_timecode(), "alreadyTC_OUT :", findItem[Column2.TC_OUT.name.lower()])
                        originalTC_OUT = otio.opentime.RationalTime.from_timecode(findItem[Column2.TC_OUT.name.lower()], fpsBase)
                        offset = originalTC_OUT - endTC
                        if offset.value <= 0: # nag value
                            issue += "end add %d\n" % -offset.value
                        else:
                            issue += "end delete %d\n" % offset.value

                # Check ShotName
                shotName = ''
                if findItem and findItem.has_key(Column2.SHOT_NAME.name.lower()):
                    shotName = findItem[Column2.SHOT_NAME.name.lower()]
                elif self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('shotName'):
                    shotName = self.cleanupTimeline[editCutInFrameList[cutIndex]]['shotName']

                if findItem and findItem.has_key(Column2.TYPE.name.lower()):
                    plateType = findItem[Column2.TYPE.name.lower()]

                # Edit Check
                try:
                    if editList[row - 1][Column2.EDIT_ORDER.name.lower()] != editOrder or editList[row - 1][Column2.CLIP_NAME.name.lower()] != clipName:
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
                        if not findItem:
                            print "new shot"
                        # 3. Omit List
                        #   Omit Shot Check Method
                        #   1) before edit to not exists new clip
                        isOmit = True
                        for newEditIndex in editCutInFrameList:
                            newClipData = self.cleanupTimeline[newEditIndex]['clip']
                            for newClipIndex, newClip in enumerate(newClipData):
                                if newClip.media_reference.is_missing_reference:  # source file missing
                                    continue
                                newClipName = os.path.basename(clip.media_reference.target_url).split('.')[0]
                                if editList[row - 1][Column2.CLIP_NAME.name.lower()] == newClipName:
                                    isOmit = False
                                    break

                            if isOmit == False:
                                break

                        if isOmit:
                            omitList.append((editList[row - 1][Column2.SHOT_NAME.name.lower()], editList[row - 1][Column2.CLIP_NAME.name.lower()]))
                            if findItem:
                                print "OmitList ? [%s]" % editList[row - 1][Column2.CLIP_NAME.name.lower()]

                        # 4. Re-Birth
                        if findItem:
                            shotInfo = getShotInfo(self.showCode, findItem[Column2.SHOT_NAME.name.lower()])
                            if shotInfo and shotInfo[0]['status'] == 'Omit':
                                print "Re-Birth", findItem[Column2.SHOT_NAME.name.lower()]
                except:
                    print len(editList), row

                # Write Data
                self.excelSheet.write(row, Column2.EDIT_ORDER.value, editOrder)
                self.excelSheet.write(row, Column2.TC_IN.value, editTCRange.start_time.to_timecode())
                self.excelSheet.write(row, Column2.TC_OUT.value, endTC.to_timecode())
                self.excelSheet.write(row, Column2.SHOT_NAME.value, shotName)
                self.excelSheet.write(row, Column2.TYPE.value, plateType)
                if findItem and findItem.has_key(Column2.VERSION.name.lower()):
                    self.excelSheet.write(row, Column2.VERSION.value, findItem[Column2.VERSION.name.lower()])

                self.excelSheet.write(row, Column2.FRAME_IN.value, frameIn)
                self.excelSheet.write(row, Column2.FRAME_OUT.value, frameOut)
                self.excelSheet.write(row, Column2.SHOT_DURATION.value, frameOut - frameIn + 1)
                self.excelSheet.write(row, Column2.ORIGINAL_ROOT_FOLDER.value, "/stuff/%s/scan" % self.showName)

                if findItem and findItem.has_key(Column2.ORIGINAL_ROOT_PATH.name.lower()):
                    self.excelSheet.write(row, Column2.ORIGINAL_ROOT_PATH.value, findItem[Column2.ORIGINAL_ROOT_PATH.name.lower()])

                if "prv" not in clipName:
                    self.excelSheet.write(row, Column2.CLIP_NAME.value, clipName)
                self.excelSheet.write(row, Column2.SCAN_DURATION.value, "%s" % editTCRange.duration.to_frames())

                in_offset = otio.opentime.RationalTime(0, 24)
                out_offset = otio.opentime.RationalTime(0, 24)
                if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('in_offset') or self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                    Msg.warning("Dissolve 있음")
                    if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('in_offset'):
                        in_offset = self.cleanupTimeline[editCutInFrameList[cutIndex]]['in_offset']
                    if self.cleanupTimeline[editCutInFrameList[cutIndex]].has_key('out_offset'):
                        out_offset = self.cleanupTimeline[editCutInFrameList[cutIndex]]['out_offset']
                    issue += "Dissolve|in_offset:%s|out_offset:%s\n" % (in_offset.value, out_offset.value)

                # if ISSUE has RETIME Info, Checked Retime Speed
                self.excelSheet.write(row, Column2.ISSUE.value, issue.strip())

                # sheet.write(row, Column2.RESOLUTION.value, "1920x1080")

                self.excelSheet.write(row, Column2.XML_NAME.value, self.editFilePath)
                self.excelSheet.write(row, Column2.MOV_CUT_IN.value, self.globalStartFrame + editCutInFrameList[cutIndex] - in_offset.value)
                try:
                    duration = editCutInFrameList[cutIndex + 1] - editCutInFrameList[cutIndex] + addMovCutDuration.to_frames()
                    self.excelSheet.write(row, Column2.MOV_CUT_DURATION.value, duration + in_offset.value + out_offset.value)
                except Exception as e:
                    duration = clip.trimmed_range().duration.to_frames()
                    self.excelSheet.write(row, Column2.MOV_CUT_DURATION.value, duration + in_offset.value + out_offset.value)
                    Msg.error(e.message)

                metadata = {'START_TC': editTCRange.start_time.to_timecode(),
                                       'END_TC': endTC.to_timecode(),
                                       'CLIPNAME': clipName}
                
                metadataList.append(metadata)
                
                if clipIndex == 0:
                    mainClipName = clipName
                    mainDuration = duration
                    mainFpsBase = fpsBase

                # if clip.media_reference.metadata.get('fcp_xml') and clip.media_reference.metadata['fcp_xml']['rate'].get('ntsc') == "TRUE":
                # ntscFps = fpsBase * (1000.0 / 1001)
                # else:
                #     ntscFps = fpsBase

                # self.excelSheet.write(row, Column2.MOV_CUT_FPS.value, "%0.2f" % ntscFps)
                self.excelSheet.write(row, Column2.MOV_CUT_FPS.value, "%0.2f" % fpsBase)

                self.excelSheet.row(row).height_mismatch = True
                self.excelSheet.row(row).height = 256 + (256 * (issue.strip().count('\n')))

                rowIndex += 1

            if hasAfterSpeedRamp:
                cutIndex += 1

            movClip = otio.schema.Clip(name=mainClipName,
                                       source_range=
                                       otio.opentime.TimeRange(otio.opentime.RationalTime(self.globalStartFrame + editCutInFrameList[cutIndex], mainFpsBase),
                                                               otio.opentime.RationalTime(mainDuration, mainFpsBase)),
                                       media_reference=mediaReference,
                                       metadata={'DEXTER': metadataList})
            movTrack.append(movClip)

            cutIndex += 1
            editOrder += 1
        print omitList


    def save(self):
        if not os.path.exists(os.path.dirname(self.excelFilePath)):
            os.makedirs(os.path.dirname(self.excelFilePath))

        self.excelData.save(self.excelFilePath)
        otio.adapters.write_to_file(self.inspectTimeline, self.inspectOtioFile)
        # otio.adapters.write_to_file(self.inspectTimeline, self.inspectOtioFile.replace('.otio', '_test.edl'))
        # otio.adapters.write_to_file(self.inspectTimeline, self.inspectOtioFile.replace('.otio', '_test.xml'))