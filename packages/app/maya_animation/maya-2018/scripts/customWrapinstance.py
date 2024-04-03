try:
    import shiboken
except:
    import shiboken2 as shiboken

from PySide2 import QtGui, QtCore, QtWidgets

def wrapinstance(ptr, base=None):
    if ptr is None:
        return None
    ptr = long(ptr)
    if globals().has_key('shiboken') or globals().has_key('shiboken2'):
        if base is None:
            qObj = shiboken.wrapInstance(long(ptr), QtCore.QObject)
            metaObj = qObj.metaObject()
            cls = metaObj.className()
            superCls = metaObj.superClass().className()
            if hasattr(QtGui,cls):
                base = getattr(QtGui, cls)
            elif hasattr(QtGui, superCls):
                base = getattr(QtGui, superCls)
            else:
                base = QtGui.QWidget
        return shiboken.wrapInstance(long(ptr), base)
    elif globals().has_key('sip'):
        base = QtCore.QObject
        return sip.wrapinstance(long(ptr), base)
    else:
        return None
