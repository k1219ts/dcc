import os
import nuke

baseDir = os.path.dirname(__file__).replace("\\", "/")
nodesDir = os.path.join(baseDir, "nodes").replace("\\", "/")

toolbar = nuke.menu("Nodes")
toolbar.removeItem("Wave Machine")
wmMenu = toolbar.addMenu("Wave Machine", icon="waveMachine.png")
def buildMenu(parentMenu, dirPath):
    for fileName in os.listdir(dirPath):
        fileStem, extension = os.path.splitext(fileName)
        filePath = os.path.join(dirPath, fileName).replace("\\", "/")
        if os.path.isdir(filePath):
            subMenu = parentMenu.addMenu(fileName)
            buildMenu(subMenu, filePath)
        else:
            nodeBaseName = os.path.splitext(fileName)[0].split("_")[0]
            iconFileName = "{}.png".format(nodeBaseName)
            command = "nuke.createNode('{}')".format(fileStem)
            if extension == ".nk":
                command = "nuke.nodePaste('{}')".format(filePath)
            parentMenu.addCommand(nodeBaseName, command, icon=iconFileName)

buildMenu(wmMenu, nodesDir)
