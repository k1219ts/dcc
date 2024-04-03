import os
from maya import cmds
import anchorUtils
import anChorMain

# ----------------------------------------------------------------------------

def divider(parent):
    line = anchorUtils.QFrame(parent)
    line.setFrameShape(anchorUtils.QFrame.HLine)
    line.setFrameShadow(anchorUtils.QFrame.Sunken)
    return line
    
# ----------------------------------------------------------------------------

class TimeInput(anchorUtils.QWidget):
    def __init__(self, parent, label, default):
        anchorUtils.QWidget.__init__(self, parent)
        
        # create layout
        layout = anchorUtils.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        
        # create label
        l = anchorUtils.QLabel(self)
        l.setText(label)
        l.setFont(anchorUtils.FONT)
        layout.addWidget(l)
        
        # create time
        self.time = anchorUtils.QSpinBox(self)
        self.time.setMinimum(0)
        self.time.setMaximum(9999)
        self.time.setValue(default)
        self.time.setFont(anchorUtils.FONT)
        layout.addWidget(self.time)
        
# ------------------------------------------------------------------------
        
    def value(self):
        return self.time.value()
    
# ----------------------------------------------------------------------------

class AnchorTransformWidget(anchorUtils.QWidget):
    def __init__(self, parent):
        anchorUtils.QWidget.__init__(self, parent)
        
        # set ui
        self.setParent(parent)        
        self.setWindowFlags(anchorUtils.Qt.Window)  

        self.setWindowTitle("Anchor Transform")      
        self.setWindowIcon(
            anchorUtils.QIcon(
                anchorUtils.findIcon("rjAnchorTransform.png")
            )
        )           
        
        self.resize(300, 100)
        
        # create layout
        layout = anchorUtils.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # time input
        self.start = TimeInput(self, "Start Frame", 1001)
        self.start.setEnabled(False)
        layout.addWidget(self.start)
        
        self.end = TimeInput(self, "End Frame", 1010)
        self.end.setEnabled(False)
        layout.addWidget(self.end)
        
        # divider
        layout.addWidget(divider(self))
        
        # create time control checkbox
        self.timeline = anchorUtils.QCheckBox(self)
        self.timeline.setChecked(True)
        self.timeline.setText("From Time Control Selection")
        self.timeline.setFont(anchorUtils.FONT)
        self.timeline.stateChanged.connect(self.setManualInputField)
        layout.addWidget(self.timeline)
        
        # divider
        layout.addWidget(divider(self))
                
        # create button
        button = anchorUtils.QPushButton(self)
        button.pressed.connect(self.doAnchor)
        button.setText("Anchor Selected Transforms")
        button.setFont(anchorUtils.FONT)
        layout.addWidget(button)
        
    # ------------------------------------------------------------------------
    
    def setManualInputField(self, state):
        self.start.setEnabled(not state)
        self.end.setEnabled(not state)
        
    # ------------------------------------------------------------------------
    
    def getFrameRangeFromTimeControl(self):
        rangeVisible = cmds.timeControl(
            anchorUtils.getMayaTimeline(), 
            q=True, 
            rangeVisible=True
        )
    
        if not rangeVisible:
            return
        
        r = cmds.timeControl(
            anchorUtils.getMayaTimeline(), 
            query=True, 
            ra=True
        )
        return [int(r[0]), int(r[-1]-1)]
        
    def getFrameRangeFromUI(self):
        start = self.start.value()
        end = self.end.value()
        
        if start >= end:
            return
            
        return [start, end]

    def getFrameRange(self):
        if self.timeline.isChecked():
            return self.getFrameRangeFromTimeControl()
        else:
            return self.getFrameRangeFromUI()
            
    # ------------------------------------------------------------------------
  
    def doAnchor(self):
        frameRange = self.getFrameRange()
        if not frameRange:
            raise ValueError("No valid frame range could be found!")
        anChorMain.anchorSelection(*frameRange)

def show():
    dialog = AnchorTransformWidget(anchorUtils.mayaWindow())
    try:
        dialog.close()
    except:
        pass
    dialog.show()

if __name__ == '__main__':
    show()