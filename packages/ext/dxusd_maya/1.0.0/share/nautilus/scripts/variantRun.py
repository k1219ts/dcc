import os
import sys
import json

nautilusFile = os.environ.get('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS')
setFile = ''
if nautilusFile:
	setFile = nautilusFile.split('\n')[0]
mainVar = setFile.replace('.mb','.json')

with open(mainVar,'r') as f:
	jsonData = json.load(f)
	version = jsonData
	print(version['mayaVersion'])




