# Copyright (c) 2012-2013 Guillaume Barlier
# This file is part of "anim_picker" and covered by the LGPLv3 or later,
# read COPYING and COPYING.LESSER for details.

from maya import OpenMayaUI

# Main Qt support
try:
    import Qt.QtWidgets as QtGui
    from Qt import QtCore, QtOpenGL
    import Qt.QtGui as QtGui2
except:
    try:
        from PySide2 import QtCore, QtGui, QtOpenGL
    except:
        raise Exception, 'Failed to import PyQt4 or Pyside'

try:
    import sip
except:
    try:
        import shiboken2
    except:
        try:
            from PySide2 import shiboken2
        except:
            raise Exception, 'Failed to import sip or shiboken'


# Instance handling
def wrap_instance(ptr, base):
    '''Return QtGui object instance based on pointer address
    '''
    if globals().has_key('sip'):
        return sip.wrapinstance(long(ptr), QtCore.QObject)
    elif globals().has_key('shiboken2'):
        return shiboken2.wrapInstance(long(ptr), base)

def unwrap_instance(qt_object):
    '''Return pointer address for qt class instance
    '''
    if globals().has_key('sip'):
        return long(sip.unwrapinstance(qt_object))
    elif globals().has_key('shiboken2'):
        return long(shiboken2.getCppPointer(qt_object)[0])


def get_maya_window():
    '''Get the maya main window as a QMainWindow instance
    '''
    try:
        ptr = OpenMayaUI.MQtUtil.mainWindow()
        return wrap_instance(long(ptr), QtGui.QMainWindow)
    except:
        #    fails at import on maya launch since ui isn't up yet
        return None
