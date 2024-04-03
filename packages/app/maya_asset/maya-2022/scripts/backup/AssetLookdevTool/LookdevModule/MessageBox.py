import Qt
import Qt.QtWidgets as QtWidgets

import maya.cmds as cmds

def MessageBox(winTitle = "Warning!",
                 Message = "Message Text",
                 Icon = "warning",
                 Button = ["OK", "Cancel"],
                 bgColor = [.5,.5,.5]):
        
        msg = cmds.confirmDialog( title=winTitle,
                              message='%s    ' % Message,
                              messageAlign='center',
                              icon=Icon,
                              button=Button,
                              backgroundColor = bgColor )
        
        return msg