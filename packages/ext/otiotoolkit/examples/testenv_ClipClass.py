import opentimelineio as otio
from core.Clip import Clip
from core.FlatternTrack import FlatternTrack

xmlFile = '/prod_nas/__DD_PROD/PRAT2/edit/210119/_org/CG_Guide/haejeok_2__Sc_087_CG_Guide_210119.xml'
movFile = '/prod_nas/__DD_PROD/PRAT2/edit/210119/_org/CG_Guide/haejeok_2__Sc_087_CG_Guide_210119.mov'

otioData = otio.adapters.read_from_file(xmlFile, 'fcp_xml')
editTimeline = FlatternTrack(otioData, movFile)
editTimeline.doIt()

for frame in editTimeline.timelineData.keys():
    if editTimeline.timelineData[frame]['clip'][0].getEffect("Retime"):
        print frame
        break

