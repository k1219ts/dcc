import mainWidget
reload(mainWidget)
from pathAnim.utils.ui import stretchFixWidget
from pathAnim.utils.ui import rebuildCurveWidget
from pathAnim.utils.ui import timeWarpWidget
from pathAnim.utils.ui import pathAnimMove

import pathAnim.utils.rebuildCurve as rebuildCurve


reload(stretchFixWidget)
reload(rebuildCurveWidget)
reload(timeWarpWidget)

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
    _win = mainWidget.PathAnimWidget()
    _win.show()
    _win.resize(220, 470)
    _win.move(2000, 500)

_sfWin = None

def showSfUI():
    global _sfWin
    if _sfWin:
        _sfWin.close()
    _sfWin = stretchFixWidget.StretchFixWidget()
    _sfWin.show()
    _sfWin.resize(300, 100)
    _sfWin.move(2000, 500)

_mvWin = None

def showMoveUI():
    global _mvWin
    if _mvWin:
        _mvWin.close()
    _mvWin = pathAnimMove.pathAnimWidget()
    _mvWin.show()
    _mvWin.resize(240, 40)
    _mvWin.move(2000, 500)

_rebuildWin = None

def showRebuildUI():
    global _rebuildWin
    if _rebuildWin:
        _rebuildWin.close()
    _rebuildWin = rebuildCurveWidget.RebuildCurveWidget()
    _rebuildWin.show()
    _rebuildWin.resize(300, 100)
    _rebuildWin.move(2000, 500)


_timewarpWin = None

def showTimewarpUI():
    global _timewarpWin
    if _timewarpWin:
        _timewarpWin.close()
    _timewarpWin = timeWarpWidget.PATimeWarpWidget()
    _timewarpWin.show()
    _timewarpWin.resize(300, 100)
    _timewarpWin.move(2000, 500)