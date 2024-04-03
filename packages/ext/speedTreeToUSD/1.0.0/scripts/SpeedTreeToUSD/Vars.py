import os
import DXUSD.Utils as utl

ROOT = utl.DirName(os.path.abspath(__file__))
ICON = '%s/ui/icons'%ROOT

class STYLE:
    RED    = 'color: rgb(250, 30, 50);'
    GREEN  = 'color: rgb(50, 250, 30);'
    GRAY   = 'color: rgb(120, 120, 120);'
    ORANGE = 'color: rgb(250, 150, 30);'
    WHITE  = 'color: white;'

class MSG:
    RESULT  = 0
    WARNING = 1
    ERROR   = 2
