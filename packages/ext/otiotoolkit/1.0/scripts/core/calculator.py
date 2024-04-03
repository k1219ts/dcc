import opentimelineio as otio
from Define import DEFAULT, STRING
from core import Tesseract
import ffmpy, subprocess, json
import Msg
import os
import OpenImageIO as oiio
import re
import sys

def getRetime(clip):
    for effect in clip.effects:
        if effect.name == "Time Remap":
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
                    if variable['value'] == DEFAULT.FALSE:
                        return False
                    else:
                        return True
    return False

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

def availableRetimeSpeed(retimeSpeed):
    return (float(retimeSpeed) < -10 or float(retimeSpeed) >= 10)

def getMovMetadata(movFile):
    result = ffmpy.FFprobe(inputs={movFile: None},
                           global_options=['-v', 'quiet',
                                           '-print_format', 'json',
                                           '-show_format', '-show_streams']
                          ).run(stdout=subprocess.PIPE)
    meta = json.loads(result[0].decode('utf-8'))
    return meta


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

                if tid == 0 or len(cleanupTimeline[frameNum]['clip']) == 0:
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
                                        print reelName, os.path.basename(afterClip.media_reference.target_url).split('.')[0]
                                        print tid, atid
                                        insertId = atid
                        except Exception as e:
                            Msg.warning(e.message)
                            insertId = cid
                        finally:
                            cleanupTimeline[frameNum]['clip'].insert(insertId, clip)
                    except Exception as e:
                        if clip.media_reference.metadata['fcp_xml']['parameter']:
                            for parm in clip.media_reference.metadata['fcp_xml']['parameter']:
                                if parm['name'] == 'Text' and parm.get('value'):
                                    clipName = parm['value']
                                    if '_' in clipName and ' ' not in clipName and len(clipName.split('_')) == 2:
                                        cleanupTimeline[frameNum]['shotName'] = clipName

                if in_offset:
                    cleanupTimeline[frameNum]['in_offset'] = in_offset
                    in_offset = None

            elif clip.schema_name() == "Transition":
                cleanupTimeline[beforeFrame]['out_offset'] = clip.out_offset
                in_offset = clip.in_offset

            elif clip.schema_name() == "Stack":
                frameNum = clip.range_in_parent().start_time.to_frames()  # + globalStartTime
                if not cleanupTimeline.has_key(frameNum):
                    cleanupTimeline[frameNum] = {'clip':[clip]}
                # Msg.warning("Has Nested Stack", dir(clip))

            elif clip.schema_name() == "Gap":
                frameNum = clip.range_in_parent().start_time.to_frames()  # + globalStartTime
                beforeFrame = frameNum
                if not cleanupTimeline.has_key(frameNum):
                    if tid == 0:
                        print frameNum
                        frameCheckList.append(frameNum)
                        cleanupTimeline[frameNum] = {'clip': []}


    return cleanupTimeline

def getTCInfo(imgFile, fps):
    print imgFile
    img = oiio.ImageInput.open(imgFile)
    if img == None:
        try:
            imgFrameNumber = imgFile.split('/')[-1].split('.')[-2]
            return otio.opentime.RationalTime(float(imgFrameNumber), fps).to_timecode()
        except Exception as e:
            return None
    attrs = img.spec().extra_attribs

    # if extension == '.dpx':
    try:
        SMPTE_TimeCode = attrs['smpte:TimeCode'][0]
    except:
        imgFrameNumber = imgFile.split('/')[-1].split('.')[-2]
        print otio.opentime.RationalTime(float(imgFrameNumber), fps).to_timecode()
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

def getTesseractTC(movFile, clipName, frame):
    tmpDir = os.tmpnam()

    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)

    startPreviewImg = os.path.join(tmpDir, '%s.%s.jpg' % (clipName, frame))
    cmd = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
    cmd += ['rvio', '-v', movFile]
    cmd += ['-t', '{FRAME}-{FRAME}'.format(FRAME=frame)]
    cmd += ['-o', startPreviewImg]

    print ' '.join(cmd)
    os.system(' '.join(cmd))

    clipAndTcList = Tesseract.getImageTCInfo(startPreviewImg)
    return clipAndTcList

    # os.system('rm -rf %s' % tmpDir)

def getTesseractClipName(movFile, clipName, frame):
    tmpDir = os.tmpnam()

    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)

    startPreviewImg = os.path.join(tmpDir, '%s.%s.jpg' % (clipName, frame))
    cmd = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
    cmd += ['rvio', '-v', movFile]
    cmd += ['-t', '{FRAME}-{FRAME}'.format(FRAME=frame)]
    cmd += ['-o', startPreviewImg]

    print ' '.join(cmd)
    os.system(' '.join(cmd))

    parseText = Tesseract.getImageTCInfo(startPreviewImg)

    clipName = ''
    splitText = parseText.split('\n')
    for text in splitText:
        convertText = text.upper().replace(' ', '').replace('O', '0')
        regex = re.compile(r'[A-Z]\d{3}[A-Z]\d{3}')
        mc = regex.findall(convertText)
        if mc:
            clipName = mc[0]
            continue
        regex = re.compile(r'[A-Z]\d{3}_[A-Z]\d{3}')
        mc = regex.findall(convertText)
        if mc:
            clipName = mc[0]
            continue

    os.system('rm -rf %s' % tmpDir)
    return clipName


def getTesseractClipNameFromTractor(movFile, clipName, frame, targetDir, index):
    regex = re.compile(r'[A-Z]\d{3}_[A-Z]\d{3}|[A-Z]\d{3}[A-Z]\d{3}')
    mc = regex.findall(clipName)
    if mc:
        clipName = mc[0] + '-'
    else:
        tmpDir = os.tmpnam()

        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)

        startPreviewImg = os.path.join(tmpDir, '%s.%s.jpg' % (clipName, frame))
        cmd = ['/backstage/dcc/DCC', 'rez-env', 'rv-7.9.2', '--']
        cmd += ['rvio', '-v', movFile]
        cmd += ['-t', '{FRAME}-{FRAME}'.format(FRAME=frame)]
        cmd += ['-o', startPreviewImg]

        print ' '.join(cmd)
        os.system(' '.join(cmd))

        parseText = Tesseract.getImageTCInfo(startPreviewImg)

        clipName = ''
        splitText = parseText.split('\n')
        for text in splitText:
            convertText = text.upper().replace(' ', '').replace('O', '0')
            regex = re.compile(r'[A-Z]\d{3}[A-Z]\d{3}')
            mc = regex.findall(convertText)
            if mc:
                for t in mc:
                    clipName += t + '-'
                continue

        if clipName == '':
            clipName = 'empty-'

        os.system('rm -rf %s' % tmpDir)

    os.system('touch %s/%s-%s-%s' % (targetDir, str(index).zfill(4), frame, clipName[:-1]))
    return clipName

def parseShowName(filePath):
    splitFilePath = filePath.split('/')
    if filePath.startswith('/prod_nas'):
        return filePath.split('/')[3].lower()
    elif 'show' in splitFilePath:
        return splitFilePath[splitFilePath.index('show') + 1]
    elif 'stuff' in splitFilePath:
        return splitFilePath[splitFilePath.index('stuff') + 1]
    else:
        assert False, "not found Show"

if __name__ == "__main__":
    # movFileName = sys.argv[1]
    # clipName = sys.argv[2]
    # frame = sys.argv[3]
    # targetDir = sys.argv[4]
    # index = sys.argv[5]
    # getTesseractClipNameFromTractor(movFileName, clipName, frame, targetDir, index)
    movFilePath = '/prod_nas/__DD_PROD/EMD/edit/20210510_2/210510_2/convert/_org/Emergency_C24_31thCGGuide_4APA_210510.mov'
    editFilePath = '/prod_nas/__DD_PROD/EMD/edit/20210510_2/210510_2/Emergency_C24_31thCGGuide_4APA_210510.xml'
    otioData = otio.adapters.read_from_file(editFilePath, 'fcp_xml')
    movMetadata = getMovMetadata(movFilePath)
    isSetMovTC = False
    movTC = '00:00:00:00'
    try:
        movTC = movMetadata['streams'][-1]['tags']['timecode']
    except Exception as e:
        isSetMovTC = True

    print movTC
    print otioData.global_start_time
    print otio.opentime.RationalTime.from_timecode(movTC, otioData.global_start_time.rate)
