import hou,dropoverlay,PySide2

def check():
    windowlist = [hou.qt.mainWindow()] + [hou.qt.floatingPanelWindow(p) 
        for p in hou.ui.curDesktop().floatingPanels()]

    if not windowlist:
        return

    inactive = [w for w in windowlist if w and not w.isActiveWindow()]

    if len(inactive) == len(windowlist):
        exists = [entry for entry in PySide2.QtWidgets.QApplication.allWidgets()
            if type(entry).__name__ == 'dragDropOverlay' and entry.isVisible()]

        if not exists:
            dropoverlay.startup(None)