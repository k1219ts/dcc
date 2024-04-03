import hou,os,sys

def open(path):
    path = hou.expandString(path)

    if not os.path.isdir(path):
        hou.ui.displayMessage("Folder doesn't exist.", 
            buttons=('OK',), severity=hou.severityType.Error, 
            default_choice=0, close_choice=0, help=None, title='Error')

        return

    platform = sys.platform

    if platform == "win32":
        path = path.replace('/','\\')
        os.startfile(path)
    elif platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])