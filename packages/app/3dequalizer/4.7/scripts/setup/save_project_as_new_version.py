#
#
# 3DE4.script.name:  5. Save Project as New Version...
#
# 3DE4.script.version:  v1.0
#
# 3DE4.script.gui:  Main Window::dx_Setup
#
# 3DE4.script.comment:  Save Project as New Version.
#
#

import os

projPath = tde4.getProjectPath()	# /show/pipe/works/MMV/shot/PKL/PKL_0290/3de/PKL_0290_main1_matchmove_v001.3de
path = os.path.dirname(projPath)	# /show/pipe/works/MMV/shot/PKL/PKL_0290/3de
file = os.path.basename(projPath)	# PKL_0290_main1_matchmove_v001.3de
fileName = file.split(".")[0]	    # PKL_0290_main1_matchmove_v001
splitFileName = fileName.split("_")	# ['PKL', '0290', 'main1', matchmove', 'v001']
cVer = splitFileName[-1]	# v001
newVer = int(cVer[1:])+1	# 001 + 1
splitFileName[-1] = "v%.3d"%newVer	# ['PKL', '0290', 'main1', matchmove', 'v002']
newFileName = "_".join(splitFileName)	# PKL_0290_main1_matchmove_v002

newProjPath = os.path.join(path, newFileName+".3de")	# /show/pipe/works/MMV/shot/PKL/PKL_0290/3de/PKL_0290_main1_matchmove_v002.3de

tde4.saveProject(newProjPath)
tde4.postQuestionRequester("Save Project as New Version.", "Saved Project.", "Ok")
