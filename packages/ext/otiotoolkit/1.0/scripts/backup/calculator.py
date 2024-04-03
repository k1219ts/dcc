import opentimelineio as otio
from Define import DEFAULT, STRING
import Msg
import os
import OpenImageIO as oiio
import dxConfig
import requests

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


def getRetime(clip):
    for effect in clip.effects:
        if effect.name == "Time Remap":
            # pprint.pprint(effect.metadata)
            for variable in effect.metadata['fcp_xml']['parameter']:
                if "speed" in variable['name']:
                    retimeSpeed = variable['value']
                    if retimeSpeed != DEFAULT.RETIME:
                        if getRetimeReverse(clip):
                            return ("-" + retimeSpeed), True
                        else:
                            return retimeSpeed, False
    return "0", False


def getRetimeReverse(clip):
    for effect in clip.effects:
        if effect.name == "Time Remap":
            for variable in effect.metadata['fcp_xml']['parameter']:
                if "reverse" in variable['name']:
                    if variable['value'] == "FALSE" or variable['value'] == "false":
                        return False
                    else:
                        return True
    return False


def getKeyframeValue2(clip, InOrOut):
    '''

    :param clip:
    :param InOrOut: STRING.RETIME_IN(speedkfin) or STRING.RETIME_OUT(speedkfout)
    :return:
    '''
    for effect in clip.effects:
        if effect.name == "Time Remap":
            for variable in effect.metadata['fcp_xml']['parameter']:
                if "graphdict" in variable['name']:
                    for keyframe in variable['keyframe']:
                        if keyframe.get(InOrOut):
                            return float(keyframe['value'])


def getTransformValue(clip):
    scale = DEFAULT.SCALE
    rotation = DEFAULT.ROTATION
    center = DEFAULT.CENTER
    anchor = DEFAULT.ANCHOR_POINT
    for effect in clip.effects:
        if effect.name == "Basic Motion":
            if isinstance(effect.metadata['fcp_xml']['parameter'], otio._otio.AnyVector):
                parameter = effect.metadata['fcp_xml']['parameter']
            else:
                parameter = [effect.metadata['fcp_xml']['parameter']]

            for variable in parameter: # effect.metadata['fcp_xml']['parameter']:
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
                elif variable['name'] == "Center":
                    center = variable['value']
                    if variable.get('keyframe'):
                        centerList = []
                        if isinstance(variable['keyframe'], otio._otio.AnyVector):
                            for keyframe in variable['keyframe']:
                                centerList.append(str(keyframe['value']))
                        else:
                            centerList.append(str(variable['keyframe']['value']))
                        center = '-'.join(centerList)
                elif variable['name'] == "Anchor Point":
                    anchor = variable['value']
                    if variable.get('keyframe'):
                        anchorList = []
                        if isinstance(variable['keyframe'], otio._otio.AnyVector):
                            for keyframe in variable['keyframe']:
                                anchorList.append(str(keyframe['value']))
                        else:
                            anchorList.append(str(variable['keyframe']['value']))
                        anchor = '-'.join(anchorList)

    return scale, rotation, center, anchor


def roundIOValue(ioValue, retimeSpeed):
    fspeed = int(float(retimeSpeed)) * 0.01
    if (ioValue / 100) > 1:
        if int(fspeed) == fspeed:  # int speed
            ioValue = round(ioValue)
        elif round(round(fspeed, 1)) > fspeed:  # find .0x
            ioValue = round(ioValue)
    else:
        ioValue = round(ioValue)  # 1xx must be round
    return ioValue


def getClipName(url):
    return os.path.basename(url).split('.')[0]

# def getInValue(clip):
#     return (clip.source_range.start_time - clip.available_range().start_time).to_frames()
#
# def getOutValue(clip):
#     return (clip.source_range.end_time_exclusive() - clip.available_range().start_time).to_frames()


def availableRetimeSpeed(retimeSpeed):
    return (float(retimeSpeed) < -10 or float(retimeSpeed) >= 10)


def cleanupTrackData(otioTimeline):
    '''
    cleanup => Main, Src setup
    :return:
    '''
    cleanupTimeline = {}
    frameCheckList = []
    for tid, track in enumerate(otioTimeline.video_tracks()):
        beforeFrame = 0
        in_offset = None
        for cid, clip in enumerate(track.each_child()):
            if clip.schema_name() == "Clip":
                frameNum = clip.range_in_parent().start_time.to_frames()  # + globalStartTime
                beforeFrame = frameNum
                if clip.media_reference.is_missing_reference:
                    if cleanupTimeline.has_key(frameNum) and cleanupTimeline[frameNum]['clip']:
                        Msg.bold(clip)
                        continue

                if clip.name == 'Bars and Tone':
                    print clip.name
                    continue

                if not cleanupTimeline.has_key(frameNum):
                    if tid == 0:
                        frameCheckList.append(frameNum)
                        cleanupTimeline[frameNum] = {'clip': []}
                    else:
                        minValue = 0
                        maxValue = 99999
                        for fid, frame in enumerate(frameCheckList):
                            if frame < frameNum and minValue < frame: # Find PreOrder FrameNum
                                minValue = frame
                            if frame > frameNum and maxValue > frame: # Find PostOrder FrameNum
                                maxValue = frame
                        frameNum = minValue

                if tid == 0 or (cleanupTimeline.has_key(frameNum) and len(cleanupTimeline[frameNum]['clip']) == 0):
                    cleanupTimeline[frameNum]['clip'].append(clip)
                else:
                    try:
                        reelName = os.path.basename(clip.media_reference.target_url).split('.')[0]
                        insertId = cid
                        frameIdx = frameCheckList.index(frameNum)

                        try:
                            # after Clip
                            if frameIdx + 1 < len(cleanupTimeline.keys()):
                                for atid, afterClip in enumerate(cleanupTimeline[frameCheckList[frameIdx + 1]]['clip']):
                                    if os.path.basename(afterClip.media_reference.target_url).split('.')[0] == reelName:
                                        # print reelName, os.path.basename(afterClip.media_reference.target_url).split('.')[0]
                                        # print tid, atid
                                        insertId = atid
                        except Exception as e:
                            Msg.warning(e.message)
                            insertId = cid
                        finally:
                            cleanupTimeline[frameNum]['clip'].insert(insertId, clip)
                    except Exception as e:

                        if clip.media_reference.metadata['fcp_xml'].get('parameter'):
                            if isinstance(clip.media_reference.metadata['fcp_xml']['parameter'], otio._otio.AnyVector):
                                for parm in clip.media_reference.metadata['fcp_xml']['parameter']:
                                    if parm['name'] == 'Text' and parm.get('value'):
                                        clipName = parm['value']
                                        if '_' in clipName and ' ' not in clipName and len(clipName.split('_')) == 2:
                                            cleanupTimeline[frameNum]['shotName'] = clipName
                            else:
                                parm = clip.media_reference.metadata['fcp_xml']['parameter']
                                if parm['name'] == 'Text' and parm.get('value'):
                                    clipName = parm['value']
                                    if '_' in clipName and ' ' not in clipName and len(clipName.split('_')) == 2:
                                        cleanupTimeline[frameNum]['shotName'] = clipName

                if in_offset:
                    cleanupTimeline[frameNum]['in_offset'] = in_offset
                    in_offset = None

            elif clip.schema_name() == "Transition":
                if cleanupTimeline.has_key(beforeFrame):
                    cleanupTimeline[beforeFrame]['out_offset'] = clip.out_offset
                in_offset = clip.in_offset

            elif clip.schema_name() == "Stack":
                frameNum = clip.range_in_parent().start_time.to_frames()  # + globalStartTime
                if not cleanupTimeline.has_key(frameNum):
                    cleanupTimeline[frameNum] = {'clip':[clip]}
                print clip
                Msg.warning("Has Nested Stack", dir(clip))

            elif clip.schema_name() == "Gap":
                frameNum = clip.range_in_parent().start_time.to_frames()  # + globalStartTime
                beforeFrame = frameNum
                if not cleanupTimeline.has_key(frameNum):
                    if tid == 0:
                        # print frameNum
                        frameCheckList.append(frameNum)
                        cleanupTimeline[frameNum] = {'clip': []}

    return cleanupTimeline


def getTCInfo(imgFile, fps):
    print imgFile
    img = oiio.ImageInput.open(imgFile)
    if img == None:
        try:
            print imgFile
            imgFrameNumber = imgFile.split('/')[-1].split('.')[-2]
            # print otio.opentime.RationalTime(float(imgFrameNumber), fps).to_timecode()
            print imgFrameNumber
            return otio.opentime.RationalTime(float(imgFrameNumber), fps).to_timecode()
        except Exception as e:
            return None
    attrs = img.spec().extra_attribs

    # if extension == '.dpx':
    try:
        SMPTE_TimeCode = attrs['smpte:TimeCode'][0]
    except:
        imgFrameNumber = imgFile.split('/')[-1].split('.')[-2]
        # print otio.opentime.RationalTime(float(imgFrameNumber), fps).to_timecode()
        return otio.opentime.RationalTime(float(imgFrameNumber), fps).to_timecode()

    assert isinstance(SMPTE_TimeCode, long), ('TimeCode not support type :', str(SMPTE_TimeCode))

    # parse SMPTE TimeCode
    indices = range(0, -8, -2)
    tcList = []
    hexTC = (hex(SMPTE_TimeCode))[2:-1]
    for i in indices:
        if i == 0:
            tcList.append('%02d' % int(hexTC[i-2:]))
        else:
            if hexTC[i-2:i] == '':
                tcList.append('00')
            else:
                tcList.append('%02d' % int(hexTC[i-2:i]))

    tcList.reverse()

    timecode = ':'.join(tcList)
    img.close()
    return timecode

