import maya.cmds as cmds
from LookdevMain import LookdevMain

def main():
    if cmds.window('LookdevTool', q = 1, ex = 1):
        cmds.deleteUI('LookdevTool')

    lookdevMain = LookdevMain()
    lookdevMain.show()
    
if __name__ == "__main__":
    main()