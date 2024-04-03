#
#
# 3DE4.script.name:  5. Save Project as New Version...
#
# 3DE4.script.version:  v1.0
#
# 3DE4.script.gui:  Main Window::DD_Setup
#
# 3DE4.script.comment:  Save Project as New Version.
#
#

import os

projPath = tde4.getProjectPath()	# /show/prat/shot/SHK/SHK_1780/matchmove/dev/3de/SHK_1780_main_matchmove_v01_01.3de
path = os.path.split(projPath)[0]	# /show/prat/shot/SHK/SHK_1780/matchmove/dev/3de
file = os.path.split(projPath)[1]	# SHK_1780_main_matchmove_v01_01.3de
fileName = file.split(".")[0]	# SHK_1780_main_matchmove_v01_01
splitFileName = fileName.split("_")	# ['SHK', '1780', 'main', matchmove', 'v01', '01']
cVer = splitFileName[-2]	# v01
newVer = int(cVer[1:])+1	# 01 + 1
splitFileName[-2] = "v%.2d"%newVer	# ['SHK', '1780', 'main', matchmove', 'v02', '01']
splitFileName[-1] = "01"
newFileName = "_".join(splitFileName)	# SHK_1780_main_matchmove_v02_01

newProjPath = os.path.join(path, newFileName+".3de")	# /show/prat/shot/SHK/SHK_1780/matchmove/dev/3de/SHK_1780_main_matchmove_v02_01.3de

tde4.saveProject(newProjPath)
tde4.postQuestionRequester("Save Project as New Version.", "Saved Project.", "Ok")
