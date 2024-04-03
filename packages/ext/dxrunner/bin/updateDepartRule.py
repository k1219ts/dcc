#!/bin/python

import os
import sys

currentShowList = ['cdh1', 'emd', 'prat2', 'slc', 'wdl', 'pipe', 'tmn', 'bmt', 'ncl', 'ncx', 'scan', 'ncscan', 'hsd']
# dccLaunchConfigPath = os.path.join(os.getenv("DXRUNNER_PATH"), 'scripts', 'resources', 'DCCLaunchList.config')
departRulePath = '/show/pipe/template/showFolderTemplate/_config/DepartRule.config'

for showName in currentShowList:
    if sys.argv[1] == 'all':
        pass
    elif sys.argv[1] != showName:
        continue
    configDir = os.path.join('/show', showName, '_config')
    cmd = 'cp -rf %s %s/' % (departRulePath, configDir)
    print cmd
    os.system(cmd)
