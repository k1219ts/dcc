#coding:utf-8
import opentimelineio as otio
from . import calculator
import os, re
from Define import DEFAULT, FORMAT, CLIPTYPE

class Clip():
    def __init__(self, clip, editMovFile, globalStartFrame):
        self.clip = clip
        self.clipNameType = "Role"
        self.schemaName = clip.schema_name()
        self.clipName = ''
        self.movFile = editMovFile
        self.startTime = None
        self.durationTime = None
        self.noOpticalDuration = None
        self.globalStartFrame = globalStartFrame
        self.effectDict = dict()
        self.dissolve = {'end_offset': otio.opentime.RationalTime(0, 24), # in_offset
                         'top_offset': otio.opentime.RationalTime(0, 24), # out_offset
                         'alignment': ''}
        self.issue = ''
        self.isRetime = False
        # self.getParseClipName()
        self.getDurationTime()
        self.getStartTime()

    def getParseClipName(self):
        clipName = calculator.getTesseractClipName(self.movFile, self.getClipName(), self.getFrame() + self.globalStartFrame)
        if clipName:
            self.clipName = clipName
        # print tcList

    def isMissingReference(self):
        if self.schemaName == "Clip":
            return False
            # return self.clip.media_reference.is_missing_reference
        return True

    def getFrame(self):
        return self.clip.range_in_parent().start_time.to_frames()

    def getEndFrame(self):
        return self.clip.range_in_parent().start_time.to_frames() + self.clip.duration().value

    def getFPS(self, ignore_ntsc=False):
        # if self.clip.source_range.start_time.rate:
        #     return self.clip.source_range.start_time.rate
        # elif self.isMissingReference():
        #     return None
        # else:
        if self.schemaName == 'Clip' and not ignore_ntsc and self.clip.media_reference.metadata.get('fcp_xml'):
            if self.clip.media_reference.metadata['fcp_xml'].get('rate'):
                if self.clip.media_reference.metadata['fcp_xml']['rate'].get('ntsc') == "TRUE":
                    return float(self.clip.media_reference.metadata['fcp_xml']['rate']['timebase'])
                else:
                    return self.clip.source_range.start_time.rate
            else:
                return self.clip.source_range.start_time.rate
        else:
            return self.clip.source_range.start_time.rate
                # assert False, 'not found fcp_xml metadtata'

    def getStartTime(self):
        if self.startTime:
            return self.startTime

        self.calcStartTime()
        return self.startTime

    def getDurationTime(self):
        if self.durationTime:
            return self.durationTime

        self.calcDurationTime()
        return self.durationTime

    def getTimeRange(self):
        return otio.opentime.TimeRange(self.getStartTime(), self.getDurationTime())

    def getEndTime(self):
        if self.getEffect("Retime"):
            return self.getTimeRange().end_time_exclusive()
        else:
            return self.getTimeRange().end_time_inclusive()

    def calcStartTime(self):
        fps = self.getFPS()
        if not fps:
            return

        if self.schemaName == "Gap" or self.isTextClip():
            startTime = self.clip.source_range.start_time.rescaled_to(fps)
        else:
            startTime = self.clip.visible_range().start_time.rescaled_to(fps)

        # if has Time Remap, calculator Retime
        if self.hasEffect() and self.effectDict.has_key("Retime"):
            startTime, durationTime = self.calcRetime()
            # print self.effectDict['Retime']
            # print "Has Retime"
            # tcList = calculator.getTesseractTC(self.movFile, self.getClipName(), self.getFrame() + self.globalStartFrame)
            # print tcList

        self.startTime = startTime

    def calcDurationTime(self):
        fps = self.getFPS()
        if not fps:
            return

        durationTime = otio.opentime.RationalTime(self.clip.visible_range().duration.to_frames(), fps)

        self.noOpticalDuration = durationTime
        self.durationTime = durationTime

        # if has Time Remap, calculator Retime
        if self.hasEffect() and self.effectDict.has_key("Retime"):
            startTime, durationTime = self.calcRetime()

        self.durationTime = durationTime

    def calcRetime(self):
        speed = float(self.effectDict['Retime']['speed'])
        reverse = self.effectDict['Retime']['reverse']
        speedIn = float(self.effectDict['Retime']['speedIn'])
        speedOut = float(self.effectDict['Retime']['speedOut'])
        # offset = int(self.effectDict['Retime']['offset'])

        if not reverse:
            # print speedIn, speedIn / self.clip.available_range().duration.value
            speedIn += speedIn / self.clip.available_range().duration.value
            # print speedIn
            retimeStart = self.clip.available_range().start_time.value + speedIn

            outOffset = self.dissolve['top_offset'].value
            # print speedOut, speedOut / self.clip.available_range().duration.value
            speedOut += speedOut / self.clip.available_range().duration.value
            outValue = speedOut - (int(speed) * 0.01) + ((int(speed) * 0.01) * outOffset)
            # print speedOut
            retimeEnd = self.clip.available_range().start_time.value + outValue
        else:
            speedIn -= speedIn / self.clip.available_range().duration.value
            retimeStart = self.clip.available_range().start_time.value + (speedIn * (-int(speed) * 0.01))

            retimeEnd = retimeStart - (self.clip.visible_range().duration.value - 1) * (-(int(speed) * 0.01))

        # if self.effectDict['Retime']['variableSpeed'] == '0':
        #     offset = float(speed) * 0.01
        # else:
        #     pass
        # offset = float(speed) * 0.01

        startTime = otio.opentime.RationalTime(retimeStart, self.getFPS())
        endTime = otio.opentime.RationalTime(retimeEnd, self.getFPS())
        durationTime = otio.opentime.RationalTime.duration_from_start_end_time(startTime, endTime)
        return startTime, durationTime

    def hasEffect(self):
        if self.effectDict:
            return True

        if hasattr(self.clip, 'effects'):
            for effect in self.clip.effects:
                if effect.name == "Time Remap":
                    speed, reverse, speedIn, speedOut, frameBlending, variableSpeed, offset = self.getRetimeInfo(effect)
                    if speed != DEFAULT.RETIME:
                        # HAS RETIME
                        if reverse:
                            speed = '-%s' % speed

                        if float(speed) < -10 or float(speed) >= 10:
                            self.effectDict["Retime"] = {'speed': speed,
                                                        'reverse': reverse,
                                                        'speedIn': speedIn,
                                                        'speedOut': speedOut,
                                                        'frameBlending': frameBlending,
                                                        'variableSpeed': variableSpeed,
                                                        'offset': offset}
                            self.isRetime = True
                            # print self.getClipName(), speed, frameBlending, variableSpeed
                elif effect.name == "Basic Motion":
                    scale, rotation, center, anchor = self.getTransformInfo(effect)
                    if scale != DEFAULT.SCALE:
                        self.effectDict["Scale"] = scale
                    if rotation != DEFAULT.ROTATION:
                        self.effectDict['Rotation'] = rotation
            return True
        return False

    def getRetimeInfo(self, effect):
        retimeSpeed = DEFAULT.RETIME
        isReverse = False
        speedInValue = '0'
        speedOutValue = '0'
        speedEndValue = '0'
        frameBlending = False
        variableSpeed = '0'
        availableDuration = '0'
        keyframeValueList = []
        keyframeWhenList = []
        for parameter in effect.metadata['fcp_xml']['parameter']:
            if 'reverse' == parameter['name']:
                if parameter['value'].upper() == DEFAULT.FALSE:
                    isReverse = False
                else:
                    isReverse = True
            elif 'speed' == parameter['name']:
                retimeSpeed = parameter['value']
            elif 'graphdict' == parameter['name']:
                for keyframe in parameter['keyframe']:
                    keyframeValueList.append(float(keyframe['value']))
                    keyframeWhenList.append(float(keyframe['when']))
                    if keyframe.get('speedkfin'):
                        speedInValue = keyframe['value']
                    elif keyframe.get('speedkfout'):
                        speedOutValue = keyframe['value']
                    elif keyframe.get('speedkfend'):
                        speedEndValue = keyframe['value']
                availableDuration = parameter['valuemax']
            elif 'frameblending' == parameter['name']:
                if parameter['value'].lower() != 'false':
                    frameBlending = True
            elif 'variablespeed' == parameter['name']:
                variableSpeed = parameter['value']

        offset = float(speedEndValue) - float(availableDuration)
        # print "offset :", float(speedEndValue), int(availableDuration), float(speedEndValue) - float(availableDuration)
        # print "retime count :", sorted(keyframeWhenList), sorted(keyframeValueList)
        # print retimeSpeed
        # for index, value in enumerate(keyframeWhenList):
        #     if keyframeValueList[index] == 0 or keyframeWhenList[index] == 0:
        #         print 0
        #     else:
        #         print keyframeValueList[index] / keyframeWhenList[index]
        return retimeSpeed, isReverse, speedInValue, speedOutValue, frameBlending, variableSpeed, offset
    
    def getTransformInfo(self, effect):
        scale = DEFAULT.SCALE
        rotation = DEFAULT.ROTATION
        center = DEFAULT.CENTER
        anchor = DEFAULT.ANCHOR_POINT
        if effect.metadata['fcp_xml'].get('parameter'):
            if isinstance(effect.metadata['fcp_xml']['parameter'], otio._otio.AnyVector):
                parameter = effect.metadata['fcp_xml']['parameter']
            else:
                parameter = [effect.metadata['fcp_xml']['parameter']]

            for variable in parameter:  # effect.metadata['fcp_xml']['parameter']:
                if variable['name'] == "Scale":
                    scale = variable['value']
                    if variable.get('keyframe'):
                        scaleList = []
                        if isinstance(variable['keyframe'], otio._otio.AnyVector):
                            for info in variable['keyframe']:
                                scaleList.append(info['value'])
                        else:
                            scaleList.append(variable['keyframe']['value'])
                        scale = '-'.join(scaleList)
                elif variable['name'] == "Rotation":
                    rotation = variable['value']
                    if variable.get('keyframe'):
                        rotateList = []
                        if isinstance(variable['keyframe'], otio._otio.AnyVector):
                            for keyframe in variable['keyframe']:
                                rotateList.append(keyframe['value'])
                        else:
                            rotateList.append(variable['keyframe']['value'])
                        rotation = '-'.join(rotateList)
                # elif variable['name'] == "Center":
                #     print variable
                #     center = variable['value']
                #     if variable.get('keyframe'):
                #         centerList = []
                #         if isinstance(variable['keyframe'], otio._otio.AnyVector):
                #             for keyframe in variable['keyframe']:
                #                 centerList.append(str(keyframe['value']))
                #         else:
                #             centerList.append(str(variable['keyframe']['value']))
                #         center = '-'.join(centerList)
                # elif variable['name'] == "Anchor Point":
                #     anchor = variable['value']
                #     if variable.get('keyframe'):
                #         anchorList = []
                #         if isinstance(variable['keyframe'], otio._otio.AnyVector):
                #             for keyframe in variable['keyframe']:
                #                 anchorList.append(str(keyframe['value']))
                #         else:
                #             anchorList.append(str(variable['keyframe']['value']))
                #         anchor = '-'.join(anchorList)

        return scale, rotation, center, anchor

    def isSlate(self):
        if self.schemaName != "Stack" and self.clip.media_reference.metadata.get('fcp_xml') and self.clip.media_reference.metadata['fcp_xml'].get('mediaSource'):
            if 'BarsAndTone' in self.clip.media_reference.metadata['fcp_xml']['mediaSource']:
                return True
        return False

    def getEffect(self, effectName):
        if self.effectDict:
            if self.effectDict.has_key(effectName):
                return self.effectDict[effectName]
            else:
                return None
        return None

    def getClipName(self):
        if self.clipName:
            return self.clipName
        if self.isMissingReference():
            return self.clip.name
        else:
            if self.clip.media_reference.is_missing_reference:
                try:
                    if self.clip.media_reference.name == '':
                        self.clipName = self.clip.name
                    else:
                        self.clipName = self.clip.media_reference.name
                    return self.clipName
                except Exception as e:
                    self.clipName = self.clip.name
                    print e.message
                    return self.clip.name
            else:
                try:
                    self.clipName = os.path.basename(self.clip.media_reference.target_url).split('.')[0]
                except:
                    self.clipName = self.clip.media_reference.name
                return self.clipName

    def getClipType(self):
        clipName = self.getClipName()

        if 'prev' in clipName or 'prv' in clipName:
            return CLIPTYPE.PREVIZ

        regex = re.compile(r'[A-Z]\d{3}_[A-Z]\d{3}|[A-Z]\d{3}[A-Z]\d{3}')
        mc = regex.findall(clipName)
        if mc:
            return CLIPTYPE.CLIP

        regex = re.compile(r'[a-zA-Z0-9]*_[0-9]{4}')
        mc = regex.findall(clipName)
        if mc:
            return CLIPTYPE.SHOTNAME

        return CLIPTYPE.NONE

    def isSpeedRamp(self, clip):
        if self.getClipName() != clip.getClipName():
            return False
        if self.getEffect("Retime") == None and clip.getEffect("Retime") == None:
            return False
        thisReverse = False
        othersReverse = False
        if self.getEffect("Retime"):
            thisReverse = self.getEffect("Retime")['reverse']
        if clip.getEffect("Retime"):
            othersReverse = clip.getEffect('Retime')['reverse']
        if thisReverse != othersReverse:
            return False

        retimeSpeed = 1
        if clip.getEffect("Retime"):
            tRetimeSpeed = clip.getEffect("Retime")['speed']
            tRetimeSpeed = round(float(tRetimeSpeed) * 0.01)
            if retimeSpeed < tRetimeSpeed:
                retimeSpeed = tRetimeSpeed

        if self.getEffect("Retime"):
            tRetimeSpeed = self.getEffect("Retime")['speed']
            tRetimeSpeed = round(float(tRetimeSpeed) * 0.01)
            if retimeSpeed < tRetimeSpeed:
                retimeSpeed = tRetimeSpeed
        print retimeSpeed

        if (clip.getStartTime() - self.getEndTime()).value >= retimeSpeed + 3:
            return False
        return True

    def mergeClip(self, clip):
        # print "Duration :", clip.getEndTime() - self.getStartTime()
        self.noOpticalDuration += clip.noOpticalDuration
        self.durationTime = clip.getEndTime() - self.getStartTime()

        if self.issue == '':
            for key in self.effectDict:
                if key == "Retime":
                    self.issue += FORMAT.SPEEDRAMPTC.format(RETIME=self.getEffect("Retime")['speed'],
                                                       TC_IN=self.getStartTime().to_timecode(),
                                                       TC_OUT=self.getEndTime().to_timecode())
                if key == "Scale":
                    self.issue += FORMAT.SCALE.format(SCALE=self.getEffect("Scale"))
                if key == "Rotation":
                    self.issue += FORMAT.ROTATION.format(ROTATE=self.getEffect("Rotation"))

        for key in clip.effectDict:
            if key == "Retime":
                self.issue += FORMAT.SPEEDRAMPTC.format(RETIME=clip.getEffect("Retime")['speed'],
                                                        TC_IN=clip.getStartTime().to_timecode(),
                                                        TC_OUT=clip.getEndTime().to_timecode())
                self.isRetime = True
            if key == "Scale":
                self.issue += FORMAT.SCALE.format(SCALE=clip.getEffect("Scale"))
            if key == "Rotation":
                self.issue += FORMAT.ROTATION.format(ROTATE=clip.getEffect("Rotation"))
                
    def IsRetime(self):
        return self.isRetime

    def isTextClip(self):
        return self.schemaName != "Stack" and self.clip.media_reference.available_range is None and self.clip.media_reference.name == "Text"