from rv import commands, rvtypes, extra_commands
from pymu import MuSymbol
import json
import opentimelineio as otio


# define hotKey
# ---------------------------------
KET_TOGGLE_OVERLAY = '/'
KEY_GOTO_GLOBALFRAME = 'G'
KEY_GOTO_CUT_BACKWARD = '<'
KEY_GOTO_CUT_FORWARD = '>'
# ---------------------------------


class OverlayHUDMode(rvtypes.MinorMode):
    def __init__(self):
        self.on = False
        self.switch = True
        rvtypes.MinorMode.__init__(self)
        self.init(
            "overlay_hud",
            [("key-down--%s" % KET_TOGGLE_OVERLAY, self.toggle, "/ key"),
             ("key-down--%s" % KEY_GOTO_GLOBALFRAME, self.changeGlobalFrame, "Go To GlobalFrame"),
             ("key-down--%s" % KEY_GOTO_CUT_BACKWARD, self.goToCutFrame, "Go To Cut frame"),
             ("key-down--%s" % KEY_GOTO_CUT_FORWARD, self.goToCutFrame, "Go To Cut frame"),
             ("graph-state-change", self.handleInput, "input taken"),
             ("frame-changed", self.frameChanged, "check desolve")
            ],
            None,
            None)

    def createText(self, node, value, hpos, vpos):
        commands.newProperty('%s.position' % node, commands.FloatType, 2)
        commands.newProperty('%s.color' % node, commands.FloatType, 4)
        commands.newProperty('%s.spacing' % node, commands.FloatType, 1)
        commands.newProperty('%s.size' % node, commands.FloatType, 1)
        commands.newProperty('%s.scale' % node, commands.FloatType, 1)
        commands.newProperty('%s.rotation' % node, commands.FloatType, 1)
        commands.newProperty("%s.font" % node, commands.StringType, 1)
        commands.newProperty("%s.text" % node, commands.StringType, 1)
        commands.newProperty('%s.debug' % node, commands.IntType, 1)

        commands.setFloatProperty('%s.position' % node, [ float(hpos), float(vpos) ], True)
        commands.setFloatProperty('%s.color' % node, [ 1.0, 1.0, 1.0, 1.0 ], True)
        commands.setFloatProperty('%s.spacing' % node, [ 1.0 ], True)
        commands.setFloatProperty('%s.size' % node, [ 0.003 ], True)
        commands.setFloatProperty('%s.scale' % node, [ 1.0 ], True)
        commands.setFloatProperty('%s.rotation' % node, [ 0.0 ], True)
        commands.setStringProperty("%s.font" % node, [""], True)
        commands.setStringProperty("%s.text" % node, [value], True)
        commands.setIntProperty('%s.debug' % node, [ 0 ], True)

    def menuState(self):
        if self.on:
            return commands.CheckedMenuState
        else:
            return commands.UncheckedMenuState

    def toggle(self, event):
        self.on = not self.on
        sources = commands.nodesOfType("RVSourceGroup")
        for source in sources:
            overlays = extra_commands.associatedNodes("RVOverlay",source + '_source')
            for overlay in overlays:
                vpos = -0.53
                if commands.propertyExists('%s.otio.metadata' % source):
                    mdata = commands.getStringProperty("%s.otio.metadata" % source)[0]
                    metadata = json.loads(mdata)

                    for idx, i in enumerate(metadata['DEXTER']):
                        name = i['CLIPNAME']
                        sTC = i['START_TC']
                        eTC = i['END_TC']
                        timeCode = '%s - %s' % (sTC, eTC)

                        if not commands.propertyExists('%s.text:clipname%s.position' % (overlay, idx)):
                            self.createText('%s.text:clipname%s' % (overlay, idx), name, -0.9, vpos)
                            self.createText('%s.text:tc%s' % (overlay, idx), timeCode, 0.48, vpos)
                        vpos-=0.05

                # Set the overlay vizibility
                commands.setIntProperty("%s.overlay.show" % overlay, [int(self.on)], True)

    def changeMediaToTimeCode(self, event):
        print("changeMediaToTimeCode")
        globalFrame = commands.frame()
        node = commands.sourcesAtFrame(globalFrame)[0]
        source = node.replace('_source', '')
        fps = round(commands.fps(),2)

        fromFrame = extra_commands.sourceFrame(globalFrame)

        if self.switch:
            media = '/home/kwantae.kim/VelozDownload/JIW_0050_publish_edit_v002.mov'
            self.switch = FalseKEY_GOTO_CUT_IN
        else:
            media = '/home/kwantae.kim/VelozDownload/S21.mov'
            self.switch = True

        commands.setSourceMedia(node, [media])

        startTimeCode = self.sourceTimeCode(1)
        startFrame = int(otio.opentime.from_timecode(startTimeCode, fps).to_frames())

        diff = fromFrame - startFrame + 1

        print('media:', media)
        print('startTimeCode:', startTimeCode, startFrame)
        print('frame:', diff)

        commands.setFrame(diff);

    def changeGlobalFrame(self, event):
        testInput = MuSymbol("gotoInput.frameInput")
        testInput('sourceGroup000000.ui.name', event)

    def handleInput(self, event):
        attr = "sourceGroup000000.ui.name"
        prop    = event.contents()
        if prop == attr:
            inputFrame = str(commands.getStringProperty(attr, 0, 10000)[0])
            self.setGlobalFrame(inputFrame)

        event.reject() # let others get it as well

    def setGlobalFrame(self, inputFrame):
        fps = round(commands.fps(),2)

        startTimeCode = self.sourceTimeCode(1)
        startFrame = int(otio.opentime.from_timecode(startTimeCode, fps).to_frames())

        diff = int(inputFrame) - startFrame + 1
        commands.setFrame(diff)

    def getDesolveFrames(self):
        frames = list()
        globalFrame = commands.frame()
        node = commands.sourcesAtFrame(globalFrame)[0]
        source = node.replace('_source', '')

        frames.append(commands.getIntProperty('%s.cut.in' % node)[0])
        frames.append(commands.getIntProperty('%s.cut.out' % node)[0])

        if commands.propertyExists('%s.otio.metadata' % source):
            mdata = commands.getStringProperty("%s.otio.metadata" % source)[0]
            metadata = json.loads(mdata)

            for idx, m in enumerate(metadata['DEXTER']):
                if 'CUT_IN' in m:
                    if not int(m['CUT_IN']) in frames:
                        frames.append(int(m['CUT_IN']))
                    if not int(m['CUT_OUT']) in frames:
                        frames.append(int(m['CUT_OUT']))
        return sorted(frames)

    def goToCutFrame(self, event):
        frames = self.getDesolveFrames()
        # print 'frames:', frames

        key = event.name().replace('key-down--', '')
        if KEY_GOTO_CUT_BACKWARD == key:
            # print 'KEY_GOTO_CUT_BACKWARD'
            self.backwardCutFrame(frames)
        elif KEY_GOTO_CUT_FORWARD == key:
            # print 'KEY_GOTO_CUT_FORWARD'
            self.forwardCutFrame(frames)

    def backwardCutFrame(self, list):
        frame = extra_commands.sourceFrame(commands.frame())
        if not frame == list[0]:
            for f in reversed(list):
                if frame > f:
                    # print 'before, after:', frame, f
                    self.setGlobalFrame(f)
                    break
        else:
            self.setGlobalFrame(frame-1)

    def forwardCutFrame(self, list):
        frame = extra_commands.sourceFrame(commands.frame())
        if not frame == list[-1]:
            for f in list:
                if frame < f:
                    # print 'before, after:', frame, f
                    self.setGlobalFrame(f)
                    break
        else:
            self.setGlobalFrame(frame+1)

    def frameChanged(self, event):
        globalFrame = commands.frame()
        node = commands.sourcesAtFrame(globalFrame)[0]
        source = node.replace('_source', '')

        fps = round(commands.fps(),2)
        sourceFrame = extra_commands.sourceFrame(globalFrame)
        timeCode = self.globalTimeCode(sourceFrame)

        if commands.propertyExists('%s.otio.metadata' % source):
            mdata = commands.getStringProperty("%s.otio.metadata" % source)[0]
            metadata = json.loads(mdata)

            for idx, m in enumerate(metadata['DEXTER']):
                if 'CUT_IN' in m:
                    cutIn = float(m['CUT_IN'])
                    cutOut = float(m['CUT_OUT'])

                    if commands.propertyExists('%s.otio.metadata' % source):
                        if sourceFrame >= cutIn and sourceFrame <= cutOut:
                            value = 1.0
                        else:
                            value = 0.0
                        if commands.propertyExists('#RVOverlay.text:clipname%s.scale' % idx):
                            commands.setFloatProperty('#RVOverlay.text:clipname%s.scale' % idx, [value])
                            commands.setFloatProperty('#RVOverlay.text:tc%s.scale' % idx, [value])

            # self.debugTimeCode()
        event.reject()

    def globalTimeCode(self, frame):
        if commands.fps() == 0.0: return ''

        f    = frame
        ifps = int(commands.fps() + 0.5)
        sec  = int(f / ifps)
        min  = sec / 60
        hrs  = min / 60
        frm  = int(f % ifps)

        if hrs == 0:
            timecode = '%02d:%02d:%02d' % (min, sec % 60, frm)
        else:
            timecode = '%02d:%02d:%02d:%02d' % (hrs, min % 60, sec % 60, frm)

        return timecode

    def sourceTimeCode(self, frame):
        infos = commands.metaEvaluate(frame)

        name = ''
        sourceFrame = frame
        fps = 0.0
        fileTC = ''

        if infos:
            for info in infos:
                if info['nodeType'] == 'RVFileSource':
                    name = info['node']
                    sourceFrame = info['frame']
                    break
            if name == '': return '__:__:__'
        else:
            sourceFrame = frame

        if fps == 0.0: fps = commands.fps()
        if fps == 0.0: return '__:__:__'

        f    = sourceFrame
        ifps = int(fps + 0.5)
        sec  = int(f / ifps)
        min  = sec / 60
        hrs  = min / 60
        frm  = int(f % ifps)

        if hrs == 0:
            return '%02d:%02d:%02d' % (min, sec % 60, frm)
        else:
            return '%02d:%02d:%02d:%02d' % (hrs, min % 60, sec % 60, frm)

    def debugTimeCode(self):
        if timeCode and commands.propertyExists('#RVOverlay.text:timeCode.text'):
            commands.setStringProperty("#RVOverlay.text:timeCode.text",
                ["timeCode: " + timeCode], True)
        else:
            try:
                offset = -0.02
                self.createText("#RVOverlay.text:timeCode", "timeCode: " +
                    timeCode, -0.05, 0)
            except:
                print('#RVOverlay.text:timeCode !error!')

def createMode():
    return OverlayHUDMode()
