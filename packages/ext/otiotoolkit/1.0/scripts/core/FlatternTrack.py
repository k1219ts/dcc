#coding:utf-8
import os
from core import Clip, calculator
import Msg
import re
import opentimelineio as otio

class FlatternTrack():
    def __init__(self, timeline, movFile):
        self.timeline = timeline
        self.timelineData = dict()
        self.frameCheckList = []
        self.movFile = movFile
        # try:
        #     self.globalStartFrame = self.timeline.global_start_time.to_frames()
        # except:
        #     self.globalStartFrame = 0

        frame = self.timeline.global_start_time.to_frames()
        if len(str(frame)) >= 7:
            self.globalStartFrame = otio.opentime.RationalTime.from_timecode(self.timeline.global_start_time.to_timecode(),
                                                            self.timeline.global_start_time.rate).to_frames()
        else:
            try:
                self.globalStartFrame = self.timeline.global_start_time.to_frames()
            except:
                self.globalStartFrame = 0
        print self.globalStartFrame
        self.doIt()

    def doIt(self):
        trackCount = len(self.timeline.video_tracks())
        cutIn = self.globalStartFrame
        editOut = self.timeline.duration().value + self.globalStartFrame

        clipIdxFromTrack = [0] * trackCount
        clipFromTrack = [None] * trackCount
        endDurFromTrack = [cutIn] * trackCount
        shotCutIn = cutIn
        beforeShotCutIn = shotCutIn
        shotClipList = []
        shotEndFrame = cutIn

        # print "Cut In :", cutIn, "EditOut :", editOut
        index = 0
        while cutIn < editOut:
            # print "Cut In :", cutIn
            nextCutIn = 999999999
            for trackIdx in range(0, trackCount):
                if endDurFromTrack[trackIdx] <= cutIn:
                    if len(self.timeline.video_tracks()[trackIdx]) > clipIdxFromTrack[trackIdx]:
                        schema = self.timeline.video_tracks()[trackIdx][clipIdxFromTrack[trackIdx]]
                        clipIdxFromTrack[trackIdx] += 1
                        if schema.schema_name() != "Transition":
                            clipFromTrack[trackIdx] = schema
                        else:
                            while True:
                                print trackIdx, clipIdxFromTrack[trackIdx]
                                try:
                                    schema = self.timeline.video_tracks()[trackIdx][clipIdxFromTrack[trackIdx]]
                                    clipIdxFromTrack[trackIdx] += 1
                                    if schema.schema_name() != "Transition":
                                        clipFromTrack[trackIdx] = schema
                                        break
                                except:
                                    break
                    else:
                        clipFromTrack[trackIdx] = None

            # print "Cut In :", cutIn, "Shot Cut In Frame", shotCutIn, "Shot End Frame :", shotEndFrame, shotClipList
            if shotCutIn != shotEndFrame and shotEndFrame == cutIn and shotClipList:
                for shotClip in shotClipList:
                    track = shotClip.clip.parent()
                    if not track:
                        return
                    tr = shotClip.clip.range_in_parent()

                    item = track.child_at_time(tr.start_time)
                    if isinstance(item, otio.schema.Transition): # Start Transition
                        if self.timelineData.has_key(shotCutIn):
                            # print "shotCutIn :", shotCutIn
                            self.timelineData[shotCutIn]['clip'][0].dissolve['end_offset'] = item.out_offset
                        # else:
                        elif self.timelineData.has_key(beforeShotCutIn):
                            # print "beforeShotCutIn :", beforeShotCutIn
                            self.timelineData[beforeShotCutIn]['clip'][0].dissolve['end_offset'] = item.out_offset

                        shotCutIn = tr.start_time.to_frames() + self.globalStartFrame
                        shotClip.dissolve['top_offset'] = item.in_offset
                        # shotClip.dissolve['alignment'] = item.metadata['fcp_xml']['alignment']
                    # item = track.child_at_time(tr.end_time_exclusive())
                    # if isinstance(item, otio.schema.Transition): # End Transition
                    #     shotClip.dissolve['end_offset'] = item.out_offset
                    #     shotClip.dissolve['alignment'] = item.metadata['fcp_xml']['alignment']

                    if not self.timelineData.has_key(shotCutIn):
                        self.timelineData[shotCutIn] = {'clip':[]}

                    if len(self.timelineData[shotCutIn]['clip']) > 0:
                        if self.timelineData[shotCutIn]['clip'][-1].isSpeedRamp(shotClip):
                            # print "Merge !", self.timelineData[shotCutIn]['clip'][-1].getClipName(), shotClip.getClipName()
                            self.timelineData[shotCutIn]['clip'][-1].mergeClip(shotClip)
                            continue

                    regex = re.compile(r'[A-Z]\d{3}_[A-Z]\d{3}|[A-Z]\d{3}[A-Z]\d{3}')
                    mc = regex.findall(shotClip.getClipName())
                    if mc:
                        pass
                    else:
                        regex = re.compile(r'[a-zA-Z0-9]*_[0-9]{4}')
                        mc = regex.findall(shotClip.getClipName())
                        if mc:
                            Msg.bold(mc)
                            if 'prv' in mc[0] or 'prev' in mc[0]:
                                pass
                            else:
                                self.timelineData[shotCutIn]['shotName'] = mc[0]

                    self.timelineData[shotCutIn]['clip'].append(shotClip)

                shotClipList = []
                beforeShotCutIn = shotCutIn
                shotCutIn = shotEndFrame

            for trackIdx in range(0, trackCount):
                schema = clipFromTrack[trackIdx]
                if schema == None:
                    continue

                clipEndFrame = schema.range_in_parent().start_time.to_frames() + schema.duration().value + self.globalStartFrame

                if endDurFromTrack[trackIdx] != clipEndFrame and endDurFromTrack[trackIdx] == cutIn:
                    endDurFromTrack[trackIdx] = clipEndFrame
                    if schema.schema_name() == "Clip" or schema.schema_name() == "Stack":
                        clip = Clip.Clip(schema, self.movFile, self.globalStartFrame)
                        shotClipList.append(clip)
                        if clipEndFrame > shotEndFrame: # and not isTransition:
                            shotEndFrame = clipEndFrame

                if nextCutIn > clipEndFrame:
                    nextCutIn = clipEndFrame

            cutIn = nextCutIn
            index += 1
            # if index == 5:
            #     break

        self.timelineData[shotCutIn] = {'clip':shotClipList}

    def getEditCutFrameList(self):
        return sorted(self.timelineData.keys())

    def getClipList(self, cutFrame):
        return self.timelineData[cutFrame]['clip']

    def getShotName(self, cutFrame):
        # print cutFrame
        # print self.timelineData[cutFrame]
        if self.timelineData[cutFrame].has_key('shotName'):
            return self.timelineData[cutFrame]['shotName']
        return ''

if __name__ == "__main__":
    otioData = otio.adapters.read_from_file('/prod_nas/__DD_PROD/EMD/edit/20210510_2/210510_2/Emergency_C24_31thCGGuide_4APA_210510.xml')
    FlatternTrack(otioData, '')
