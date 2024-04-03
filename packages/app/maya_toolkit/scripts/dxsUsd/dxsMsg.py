import string

try:
    import maya.api.OpenMaya as OpenMaya
    import maya.cmds as cmds
except:
    pass

def Print(state, message):
    try:
        if cmds.about(batch=True) and state == 'dialog':
            state = 'warning'

        if state == 'dialog':
            msg = message
            if type(message).__name__ == 'list':
                msg = string.join(message, '\n')
            cfm = cmds.confirmDialog(m=msg, title='Warning', icon='warning', b=['continue', 'stop'], cb='stop')
            if cfm == 'stop':
                message.insert(0, '')
                cmds.waitCursor(state=False)
                assert False, string.join(message, '\n')
        else:
            if type(message).__name__ == 'list':
                for m in message:
                    _print(state, m)
            else:
                _print(state, message)
    except NameError as e:
        print e.message

def _print(state, message):
    if state == 'warning':
        OpenMaya.MGlobal.displayWarning(message)
    elif state == 'error':
        OpenMaya.MGlobal.displayError(message)
    else:
        OpenMaya.MGlobal.displayInfo('Info: %s' % message)
