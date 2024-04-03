import opentimelineio as otio
import os

filePath = '/prod_nas/__DD_PROD/SLC/edit/20201111/S30_MET/S#30_MET_v1_CGDI_201111.fcpxml'
print filePath
otioTimeline = otio.adapters.read_from_file(filePath, 'fcpx_xml')

# print dir(otioTimeline[0])
# print otioTimeline[0].duration()

for clip in otioTimeline[0].video_tracks()[0].each_clip():
    print clip.name
    print clip.range_in_parent().start_time.to_frames()
    print clip.visible_range()
    print