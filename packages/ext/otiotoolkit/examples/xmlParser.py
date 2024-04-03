#coding:utf-8
import opentimelineio as otio
import sys
import os
import xlwt, xlrd2
from enum import Enum, unique

@unique
class Column(Enum):
    # Excel Column Define
    EDIT_ORDER = 0
    SHOT_NAME = 1
    PLATE_TYPE = 2
    PLATE_VERSION = 3
    FRAME_IN = 4
    FRAME_OUT = 5
    RETIME = 6
    SCALE = 7
    ORIGINAL_ROOT_FOLDER = 8
    ORIGINAL_ROOT_PATH = 9
    EXR_FILENAME = 10
    START_TC = 11
    END_TC = 12
    SHOT_DURATION = 13
    PLATE_RESOLUTION = 14
    SCAN_DURATION = 15
    ISSUE = 16
    XML_NAME = 17
    MOV_START_TC = 18
    MOV_DURATION = 19

# filename = sys.argv[1]
filename = '/show/pipe/template/EditorialPipeline/data/PIPE/fcp7_xml.xml'
showName = 'pipe'

otioData = otio.adapters.read_from_file(filename)
# print otioData.global_start_time
mainTrack = otioData.video_tracks()[0]

excelFileName = '/show/{SHOW}/_config/EDIT/plate_list.xls'.format(SHOW=showName)

excel = xlwt.Workbook(encoding='utf-8')
sheet = excel.add_sheet("scan_list")

alreadyRows = 0
if os.path.exists(excelFileName):
    plateListExcel = xlrd2.open_workbook(excelFileName)
    if plateListExcel.sheet_by_name('scan_list').nrows > 0:
        alreadyRows = plateListExcel.sheet_by_name('scan_list').nrows
        for row in range(0, plateListExcel.sheet_by_name('scan_list').nrows):
            for col in range(0, plateListExcel.sheet_by_name('scan_list').ncols):
                sheet.write(row, col, plateListExcel.sheet_by_name('scan_list').row_values(row)[col])
else:
    sheet.write(0, Column.EDIT_ORDER.value, "edit_order")
    sheet.write(0, Column.SHOT_NAME.value, "shot_name")
    sheet.write(0, Column.PLATE_TYPE.value, "type")
    sheet.write(0, Column.PLATE_VERSION.value, "version")
    sheet.write(0, Column.FRAME_IN.value, "frame_in")
    sheet.write(0, Column.FRAME_OUT.value, "frame_out")
    sheet.write(0, Column.RETIME.value, "retime")
    sheet.write(0, Column.SCALE.value, "scale")
    sheet.write(0, Column.ORIGINAL_ROOT_FOLDER.value, "original_root_folder")
    sheet.write(0, Column.ORIGINAL_ROOT_PATH.value, "original_root_path")
    sheet.write(0, Column.EXR_FILENAME.value, "exr_filename")
    sheet.write(0, Column.START_TC.value, "tc_in")
    sheet.write(0, Column.END_TC.value, "tc_out")
    sheet.write(0, Column.SHOT_DURATION.value, "shot_duration")
    sheet.write(0, Column.PLATE_RESOLUTION.value, "resolution")
    sheet.write(0, Column.SCAN_DURATION.value, "scan_duration")
    sheet.write(0, Column.ISSUE.value, "issue")
    sheet.write(0, Column.XML_NAME.value, "xml_name")
    alreadyRows = 1


# index = 0
# rowIndex = 0
# while index < len(mainTrack):
#     clip = mainTrack[index]
#     print clip.name, clip.schema_name()
#     # print clip.schema_name(), clip.visible_range().start_time.to_timecode(), clip.visible_range().duration, clip.name, clip.media_reference.is_missing_reference
#     # row = rowIndex + alreadyRows
#     # editOrder = row
#     # plateType = "main1"
#     # plateVersion = "v001"
#     # frameIn = 1001
#     # frameOut = 1001 + clip.visible_range().duration.to_frames()
#     # movFile = clip.media_reference.target_url
#
#     retimeSpeed = 0
#     if clip.schema_name() == "Clip" and len(clip.effects) != 0:
#         for effect in clip.effects:
#             print "Effect :", effect.effect_name
#             # if "Time Remap" in effect.name:
#             #     for variable in effect.metadata['fcp_xml']['parameter']:
#             #         if "speed" in variable['name']:
#             #             retimeSpeed = variable['value']
#             #             print "Retime :", retimeSpeed
#             #             break
#
#     # if retimeSpeed != 0: # has retime
#     #     beforeClip = mainTrack[index - 1]
#     #     afterClip = mainTrack[index + 1]
#     #     beforeFileName = os.path.basename(beforeClip.media_reference.target_url).split('.')[0]
#     #     currentFileName = os.path.basename(clip.media_reference.target_url).split('.')[0]
#     #     afterFileName = os.path.basename(afterClip.media_reference.target_url).split('.')[0]
#     #
#     #     isSpeedRamp = False
#     #     if beforeFileName == currentFileName:
#     #         isSpeedRamp = True
#     #         print "Same Before"
#     #     elif afterFileName == currentFileName:
#     #         isSpeedRamp = True
#     #         print "Same After"
#     #         index += 1
#     #     else: # All Retime
#     #         pass
#     #
#     #     if isSpeedRamp:
#     #         sheet.write(row, Column.RETIME.value, "SpeedRamp {RETIME}%".format(RETIME=retimeSpeed))
#     #         sheet.write(row, Column.ISSUE.value, "SpeedRamp TC정보: %s - %s" % (clip.visible_range().start_time.to_timecode(), (clip.visible_range().start_time + clip.visible_range().duration).to_timecode()))
#     #     else:
#     #         sheet.write(row, Column.RETIME.value, "Retime {RETIME}%".format(RETIME=retimeSpeed))
#     #
#     #
#     # sheet.write(row, Column.EDIT_ORDER.value, editOrder)
#     # sheet.write(row, Column.PLATE_TYPE.value, "main1")
#     # sheet.write(row, Column.PLATE_VERSION.value, "v001")
#     # sheet.write(row, Column.FRAME_IN.value, 1001)
#     # sheet.write(row, Column.FRAME_OUT.value, 1001 + clip.visible_range().duration.to_frames())
#     # if not clip.media_reference.is_missing_reference:
#     #     sheet.write(row, Column.EXR_FILENAME.value, os.path.basename(clip.media_reference.target_url).split('.')[0])
#     # sheet.write(row, Column.START_TC.value, clip.visible_range().start_time.to_timecode())
#     # sheet.write(row, Column.END_TC.value, (clip.visible_range().start_time + clip.visible_range().duration).to_timecode())
#     # sheet.write(row, Column.SCAN_DURATION.value, clip.visible_range().duration.to_frames())
#     # sheet.write(row, Column.XML_NAME.value, filename)
#
#     index += 1
#     rowIndex += 1

# print dir(clip)

if not os.path.exists(os.path.dirname(excelFileName)):
    os.makedirs(os.path.dirname(excelFileName))

excel.save(excelFileName)