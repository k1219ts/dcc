import mainWidget
reload(mainWidget)

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        # _win.deleteLater()
    _win = mainWidget.PathRiggingWidget()
    _win.show()
    _win.resize(200, 100)