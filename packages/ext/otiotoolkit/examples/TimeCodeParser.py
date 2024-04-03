'''
exec cmd : DCC.local dev otiotoolkit -- python TimeCodeParser.py
'''
import glob
import os
import opentimelineio as otio
import OpenImageIO as oiio

checkPlateExtension = 'exr'
# plateDir = os.path.join('/show/pipe/template/otiotoolkit', 'data', 'plates', checkPlateExtension)
# plateDir = '/stuff/emd/scan/20201204/201203_vfx/Emergency_A01_PreCG_Source_CAR_201130/019_B096C009_200620_R3MY'
# plateDir = '/stuff/prat2/scan/201210/prat2_2K_request_20201209/002_W003_C003_01013A'
# plateDir = '/stuff/slc/scan/201224_vfx/S39_DAT_v2_nooptial_201222/003_B076C003_201027_RNCA'
# plateDir = '/stuff/prat2/scan/201229/prat2_2K_request_20201229/001_W005_C007_0101EK'
# plateDir = '/stuff/emd/scan/20210114/210114_vfx/Emergency_A_dir_3rd_CGSource_LAD_210111/019_B394C017_200918_R3MY'
# plateDir = '/stuff/emd/scan/20210118/210114_29.97/Rat_all/D003C001'
# plateDir = '/stuff/emd/scan/20210125/210125_vfx/Emergency C07_10th_CGSource_210123/004_P003C004'
# plateDir = '/stuff/prat2/scan/210126/prat2_S87B_2k_request_20210122/068_C210C019_210106_C46Q'
# plateDir = '/stuff/emd/scan/20210127/210126_vfx/Emergency_C07_9th_CGSource_NRT_210122/173_R003_C002_1224XR'
# plateDir = '/stuff/emd/scan/20210202/210202_vfx/Emergency_C09_12th_CGSource_210126/266_A519C009_201014_R4AK'
# plateDir = '/stuff/emd/scan/20210209/210208_vfx/Emergency_17th_CGSource_210208/003_A260C017_200802_R4AK/'
plateDir = '/stuff/slc/scan/210512_vfx/SLC_CG_list_01_0504_DI/001_A203C003_210206_R27N'

plateImageList = sorted(glob.glob('%s/*' % plateDir))

extension = os.path.splitext(plateImageList[0])[-1]
plateStartIndex = 0
dur = 49
VFXShotName = "OTI_0010"
FrameStartTime = 1001

def getTCInfo(imgFile):
    img = oiio.ImageInput.open(imgFile)
    attrs = img.spec().extra_attribs

    # if extension == '.dpx':
    SMPTE_TimeCode = attrs['smpte:TimeCode'][0]

    assert isinstance(SMPTE_TimeCode, long), ('TimeCode not support type :', str(SMPTE_TimeCode))

    # parse SMPTE TimeCode
    indices = range(0, -8, -2)
    tcList = []
    hexTC = (hex(SMPTE_TimeCode))[2:-1]
    # print hexTC
    for i in indices:
        if i == 0:
            tcList.append('%02d' % int(hexTC[i-2:]))
        else:
            try:
                tcList.append('%02d' % int(hexTC[i-2:i]))
            except:
                tcList.append('00')

    tcList.reverse()

    timecode = ':'.join(tcList)
    return timecode

# tcList = []
# for plateImg in plateImageList:
#     print plateImg
#     tc = getTCInfo(plateImg)
#     if tc in tcList:
#         print "????"
#     tcList.append(tc)

print plateImageList[0], plateImageList[-1]
startTimeCode = getTCInfo(plateImageList[0])
endTimeCode= getTCInfo(plateImageList[-1])
# #
plateStartTime = otio.opentime.from_timecode(startTimeCode, 24)
plateEndTime = otio.opentime.from_timecode(endTimeCode, 24)
#
# plateRange = otio.opentime.TimeRange(plateStartTime, plateEndTime - plateStartTime)
#
# print otio.opentime.RationalTime(446, 24).to_timecode()

print otio.opentime.RationalTime(19179, 23.976).to_timecode()
# print otio.opentime.RationalTime(381.48, 23.976).rescaled_to(24.0)

# print (otio.opentime.from_timecode('01:36:14:19', 24.0) - otio.opentime.from_timecode('01:35:56:10', 24.0)).value

# retimeStartTC = otio.opentime.RationalTime(899895, 24) + otio.opentime.RationalTime(168, 24)
# retimeEndTC = otio.opentime.RationalTime(899895, 24) + otio.opentime.RationalTime(264.997, 24)
# retimeRange = otio.opentime.TimeRange(plateStartTime, retimeEndTC - retimeStartTC)
# print retimeStartTC.to_timecode(), retimeEndTC.to_timecode(), retimeRange.duration
#
# otioData = otio.adapters.read_from_file('/show/pipe/template/EditorialPipeline/data/PIPE/fcp7_xml.xml', 'fcp_xml')
# retimeClip = None
#
# for clip in otioData.video_tracks()[0].each_clip():
#     if clip.effects:
#         retimeClip = clip
#         break
#
# print retimeClip.visible_range().start_time.to_timecode()
# print retimeClip.visible_range().end_time_inclusive().to_timecode()
# print retimeClip.visible_range().duration

print plateStartTime.to_timecode(), plateEndTime.to_timecode()

# print plateRange.end_time_exclusive().to_timecode()
# print plateRange.end_time_inclusive().to_timecode()
# print (plateRange.end_time_exclusive() - otio.opentime.RationalTime(1, 24)).to_timecode()
#
#
# print plateRange.duration
# editStartTime = otio.opentime.from_timecode('20:19:19:10', 24)
# editEndTime = otio.opentime.from_timecode('20:19:22:06', 24)
# editRange = otio.opentime.TimeRange(editStartTime, editEndTime - editStartTime)
#
# print editRange.duration

# print otio.opentime.RationalTime(88299, 30).to_timecode()
# print otio.opentime.RationalTime(131199, 30).to_timecode()

# print 24 * 1000.0 / 1001
# temp = otio.opentime.from_timecode("00:32:21:11", 24)
# print (temp + otio.opentime.RationalTime(200, 24)).to_timecode()
# real = otio.opentime.from_timecode("03:59:04:21", 24)
# print temp - real

# print otio.opentime.from_frames(49)

# opentime = otio.opentime.from_frames(100, 24)
# print opentime.rescaled_to(23.98)
# print dir(plateEndTime)
# print otio.opentime.RationalTime.duration_from_start_end_time()
# print dir(otio.opentime.TimeRange(plateStartTime, plateEndTime - plateStartTime))

# print otio.opentime.from_frames(1239108, 24).to_timecode()
# print plateStartTime.to_timecode(), plateEndTime.to_timecode()
# hrs, mins, secs, frs = [hex(SMPTE_TimeCode)[i:i+2] for i in indices]
# print hrs, mins, secs, frs

# elif extension == '.exr':
#     frames = attrs['smpte:TimeCode'][0]
#     openPlateTime = otio.opentime.from_frames(frames, 24)

# editStartTCSample1 = '02:14:06:05'
# openEditTime1 = otio.opentime.from_timecode(editStartTCSample1, 24)
#
# editStartTCSample2 = '02:14:06:08'
# openEditTime2 = otio.opentime.from_timecode(editStartTCSample2, 24)
#
# editStartTCSample3 = '02:14:06:10'
# openEditTime3 = otio.opentime.from_timecode(editStartTCSample3, 24)
#
#
# if openPlateTime == openEditTime2:
#     plateStartIndex = 0
#     shotPlateList = plateImageList[plateStartIndex:plateStartIndex + dur + 1]
#     convertPlateList = []
#     for index, shotPlate in enumerate(shotPlateList):
#         convertPlateList.append((shotPlate, '%s.%s.dpx' % (VFXShotName, FrameStartTime + index)))
#         print convertPlateList[-1]
#
#     assert dur == len(convertPlateList), "mismatch duration : duration is '%d' but images count '%d'" % (dur, len(convertPlateList))
#
# print '#' * 80
#
# if openPlateTime < openEditTime3:
#     plateStartIndex = openEditTime3.to_frames() - openPlateTime.to_frames()
#     shotPlateList = plateImageList[plateStartIndex:plateStartIndex + dur + 1]
#     convertPlateList = []
#     for index, shotPlate in enumerate(shotPlateList):
#         convertPlateList.append((shotPlate, '%s.%s.dpx' % (VFXShotName, FrameStartTime + index)))
#         print convertPlateList[-1]
#     assert dur == len(convertPlateList), "mismatch duration : duration is '%d' but images count '%d'" % (dur, len(convertPlateList))
#
# assert not (openPlateTime > openEditTime1), "[plateStartImageTC < editStartTC] : not found matching edit TC to plate image"
