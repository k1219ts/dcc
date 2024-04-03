import opentimelineio as otio
import os

filePath = '/prod_nas/__DD_PROD/CDH/edit/during_pre/201123_keyshot/Untitled_Project_(Resolve).xml'

otioTimeline = otio.adapters.read_from_file(filePath)
otioTimeline.global_start_time = otio.opentime.RationalTime(60 * 60 * 24, 24)
print otioTimeline.global_start_time

for clip in otioTimeline.each_clip():
    # print clip.media_reference
    print clip.trimmed_range().start_time.to_timecode(), clip.trimmed_range().start_time.to_frames()
    print clip.visible_range().start_time.to_timecode(), clip.visible_range().start_time.to_frames()
    print clip.trimmed_range_in_parent().start_time.to_timecode()
    if hasattr(clip, "effects"):
        while len(clip.effects) != 0:
            clip.effects.pop()
        # print clip.trimmed_range().start_time.to_timecode()
        # for effect in clip.effects:
        #     print effect
        #     if effect.name == "Basic Motion":
        #         for variable in effect.metadata['fcp_xml']['parameter']:
        #             if "Scale" in variable['name']:
        #                 scale = variable['value']
        #                 print scale
        #             elif 'Rotation' in variable['name']:
        #                 rotation = variable['value']
        #                 print rotation
        #             elif 'Center' in variable['name']:
        #                 print variable
        #                 # center = variable['value']
        #                 # print center
        #             elif 'Anchor Point' in variable['name']:
        #                 anchorPoint = variable['value']
        #                 print anchorPoint
        #
        #     elif effect.name == "Time Remap":
        #         for variable in effect.metadata['fcp_xml']['parameter']:
        #             if "speed" in variable['name']:
        #                 retimeSpeed = variable['value']
        #                 if retimeSpeed != '0':
        #                     print retimeSpeed

    # try:
    #     plateMovName = os.path.basename(clip.media_reference.target_url)
    #     TCInfo = clip.visible_range()
    #     print plateMovName, TCInfo.start_time.time_code(), TCInfo.start_time.duration
    # except:
    #     print clip

print
print '*' * 80
print

# filePath = '/prod_nas/__DD_PROD/CDH/edit/during_pre/200519/OPT/0519_Alien_s#41C_OPT.edl'
#
# otioTimeline = otio.adapters.read_from_file(filePath)
# otioTimeline.global_start_time = otio.opentime.RationalTime(60 * 60 * 24, 24)
# print otioTimeline.global_start_time
#
# for clip in otioTimeline.each_clip():
#     # print clip.media_reference
#     print clip.trimmed_range().start_time.to_timecode()
#     print clip.visible_range().start_time.to_timecode()
#     print clip.trimmed_range_in_parent().start_time.to_timecode()