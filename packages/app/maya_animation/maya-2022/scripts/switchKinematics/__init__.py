import mainwidget
reload(mainwidget)

_win = None


def showUI():
    global _win
    if _win:
        _win.close()
        # _win.deleteLater()
    _win = mainwidget.Fk2Ik()
    _win.show()
    _win.resize(200, 100)
