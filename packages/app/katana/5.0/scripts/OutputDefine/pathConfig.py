import os
import string
import json

def GetProjectPath(show=None, scenePath=None):
    showName = show
    showDir  = ''

    if show:
        showDir = '/show/{NAME}'.format(NAME=showName)

    if scenePath and scenePath.find('/show/') > -1:
        splitPath = scenePath.split('/')
        showIndex = splitPath.index('show')
        showPath  = string.join(splitPath[:showIndex+1], '/')
        if not showName:
            showName = splitPath[showIndex+1]
        showDir = os.path.join(showPath, showName)

    pathRuleFile = '{SHOWDIR}/_config/maya/pathRule.json'.format(SHOWDIR=showDir)
    if os.path.exists(pathRuleFile):
        try:
            ruleData = json.load(open(pathRuleFile))
            if ruleData.has_key('showDir') and ruleData['showDir']:
                __showDir = ruleData['showDir']
                if os.path.isabs(__showDir):
                    showDir = __showDir
                else:
                    showDir = os.path.join(os.path.dirname(showDir), __showDir)
        except:
            pass
    # else:
    #     if showDir.find('_pub') == -1:
    #         showDir += '_pub'

    assert showName, '# msg: not found showName'
    assert showDir,  '# msg: not found showDir'
    return str(showDir), str(showName)
