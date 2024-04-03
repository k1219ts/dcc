import maya.cmds as cmds


def MessageBox(Message="Message Text",
               winTitle="Warning!",
               Icon="warning",
               Button=["OK"],
               bgColor=[1, 1, 1]):
    msg = cmds.confirmDialog(title=winTitle,
                             message='%s    ' % Message,
                             messageAlign='center',
                             icon=Icon,
                             button=Button,
                             backgroundColor=bgColor)

    return msg