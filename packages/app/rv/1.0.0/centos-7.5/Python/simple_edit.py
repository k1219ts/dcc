import pprint
from rv import rvtypes, commands, extra_commands

INTMAX = 2147483647

class SimpleEdit(rvtypes.MinorMode):

    #---------------------------------------------------------------------------
    def updateFrame(self):
        self.m_gframe = commands.frame()
        self.m_sframe = extra_commands.sourceFrame(self.m_gframe, None)
    def getSource(self):
        self.updateFrame()
        sources = commands.sourcesAtFrame(self.m_gframe)
        if sources:
            return sources[0]

    #---------------------------------------------------------------------------
    # SET
    def setCutIn(self, node, frame):
        commands.setIntProperty(node + ".cut.in", [frame], True)
    def setCutOut(self, node, frame):
        commands.setIntProperty(node + ".cut.out", [frame], True)

    #---------------------------------------------------------------------------
    # GET
    def getCutIn(self, node):
        return commands.getIntProperty(node + ".cut.in")[0]
    def getCutOut(self, node):
        return commands.getIntProperty(node + ".cut.out")[0]


    #---------------------------------------------------------------------------
    # Source Cut-In
    def frameCutIn(self, event):
        node = self.getSource()
        if node:
            reset_frame = self.m_gframe - self.m_sframe + 1
            current_cutin = self.getCutIn(node)
            if abs(current_cutin) != INTMAX:
                reset_frame += current_cutin - 1
            self.setCutIn(node, self.m_sframe)
            commands.setFrame(reset_frame)
    # Source Cut-Out
    def frameCutOut(self, event):
        node = self.getSource()
        if node:
            self.setCutOut(node, self.m_sframe)


    #---------------------------------------------------------------------------
    # Reset
    def getResetFrame(self, node):
        current_cutin = self.getCutIn(node)
        if abs(current_cutin) != INTMAX:
            return self.m_gframe + current_cutin - 1

    def resetCutIn(self, event):
        node = self.getSource()
        if node:
            reset_frame = self.getResetFrame(node)
            self.setCutIn(node, -INTMAX)
            if reset_frame:
                commands.setFrame(reset_frame)

    def resetCutOut(self, event):
        node = self.getSource()
        if node:
            self.setCutOut(node, INTMAX)

    def resetCutInOut(self, event):
        node = self.getSource()
        if node:
            reset_frame = self.getResetFrame(node)
            self.setCutIn(node, -INTMAX)
            self.setCutOut(node, INTMAX)
            if reset_frame:
                commands.setFrame(reset_frame)


    #---------------------------------------------------------------------------
    def queryInfo(self, event):
        node = self.getSource()
        if node:
            cutin = commands.getIntProperty(node + ".cut.in")
            cutout= commands.getIntProperty(node + ".cut.out")
            print(cutin, cutout)


    def __init__(self):
        rvtypes.MinorMode.__init__(self)

        self.m_gframe = None
        self.m_sframe = None

        self.init("simple_edit",
            [
                ("key-down--control--[", self.frameCutIn, "Source Cut-In"),
                ("key-down--control--]", self.frameCutOut, "Source Cut-Out"),
                ("key-down--control--{", self.resetCutIn, "Source Cut-In Reset"),
                ("key-down--control--}", self.resetCutOut, "Source Cut-Out Reset"),
                ("key-down--control--backspace", self.resetCutInOut, "Source Cut-In-Out Reset"),
                # ("key-down--Z", self.queryInfo, "Source Cut info")
            ],
            None
        )


def createMode():
    "Required to initialize the module. RV will call this function to create your mode."
    return SimpleEdit()
