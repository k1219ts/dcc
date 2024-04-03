import os
import re
import json
import string


def GetLastVersion(dirpath):
    lastversion = 'v001'
    if os.path.exists(dirpath):
        versions = list()
        for d in os.listdir(dirpath):
            if not d.startswith('.') and os.path.isdir(os.path.join(dirpath, d)):
                regex = re.compile('v\d\d\d').findall(d)
                if regex and regex[0] == d:
                    versions.append(d)
        versions.sort()
        if versions:
            lastversion = versions[-1]
    return lastversion

def GetVersion(dirPath, overWrite=True):
    if not os.path.exists(dirPath):
        return 'v001'

    last = GetLastVersion(dirPath)
    if overWrite:
        return 'v%03d' % (int(last[1:]) + 1)
    else:
        return last


# def GetRelPath(current, target):
#     if target.startswith('.'):
#         return target
#     if target.startswith('/assetlib'):
#         return target
#
#     currentDir = current
#     baseName   = os.path.basename(current)
#     if len(baseName.split('.')) > 1:    # current is file
#         currentDir = os.path.dirname(current)
#
#     relfile = os.path.relpath(target, start=currentDir)
#     if relfile[0] != '.':
#         relfile = './' + relfile
#     return relfile
def GetRelPath(current, target):
    '''
    Args
        current : file or directory
        target  : file
    '''
    comprefix = os.path.commonprefix([current, target])
    if comprefix == '' or comprefix == '/' or comprefix == '/show/':
        return target

    curdir = current
    basenm = os.path.basename(current)
    if basenm:
        if len(basenm.split('.')) > 1:  # is file
            curdir = os.path.dirname(current)
    else:   # is end string include '/'
        curdir = os.path.dirname(current)
    # print '# CurrentDir :', curdir
    tardir = os.path.dirname(target)
    if curdir == tardir:
        return './' + os.path.basename(target)
    else:
        rel = os.path.relpath(target, start=curdir)
        if rel[0] != '.':
            rel = './' + rel
        return rel



def GetProjectPath(show=None, maya=None):
    '''
    Output directory compute by pathRule.json
    Args:
        show (str): showName
        maya (str): maya filePath
    Returns:
        showDir (str):
        showName(str):
    '''
    showDir = None; showName = None;
    if show:
        if "/show/" in show:
            showDir = show
        else:
            showDir = '/show/{NAME}'.format(NAME=show)
        showDir = GetOutShowDir(showDir)
        showName= show.replace('_pub', '')

    if maya:
        maya = maya.replace('/netapp/dexter/show', '/show')
        maya = maya.replace('/space/dexter/show', '/show')
        if maya.find('/show/') > -1:
            splitPath = maya.split('/')
            showIndex = splitPath.index('show')
            showName  = splitPath[showIndex+1]
            showDir   = string.join(splitPath[:showIndex+2], '/')
            showDir = GetOutShowDir(showDir)
            showName= showName.replace('_pub', '')
        elif maya.find('/assetlib/') > -1:
            splitPath = maya.split('/')
            showDir = '/assetlib/3D'
            showName= '3D'
    return str(showDir), str(showName)

def GetOutShowDir(showDir):
    rootPath = os.path.dirname(showDir)
    rule = GetPathRule(showDir)
    if rule.has_key('showDir') and rule['showDir']:
        dir = rule['showDir']
        if os.path.isabs(dir):
            showDir = dir
        else:
            showDir = os.path.join(rootPath, dir)
    return showDir

def GetPathRule(showDir):
    dir = showDir.replace('_pub', '')
    ruleFile = '{DIR}/_config/maya/pathRule.json'.format(DIR=dir)
    result   = dict()
    if os.path.exists(ruleFile):
        result = json.load(open(ruleFile))
    return result


#-------------------------------------------------------------------------------
def GetRootPath(dirpath):
    '''
    Output show directory compute by pathRule.json
    Returns
        showDir, showName
    '''
    splitPath = dirpath.split('/')

    if "assetlib" in splitPath:
        sid = splitPath.index("assetlib")
        showname = splitPath[sid + 1]
        showdir = '/'.join(splitPath[:sid + 2])
        return showdir, showname

    sid = splitPath.index('show')
    showname= splitPath[sid + 1]
    showdir = '/'.join(splitPath[:sid + 2])

    rootName= showname.replace('_pub', '')
    rootDir = showdir

    rule = GetConfig(showdir, 'pathRule.json')
    if rule:
        if rule.has_key('showDir') and rule['showDir']:
            dir = rule['showDir']
            if os.path.isabs(dir):
                rootDir = dir
            else:
                rootDir = os.path.join(os.path.dirname(showdir), dir)
    # print '>>', rootDir
    return rootDir, rootName


#-------------------------------------------------------------------------------

def GetMayaConfigPath(showDir):
    dir = showDir.replace('_pub', '')
    return os.path.join(dir, '_config', 'maya')

def GetConfig(showDir, filename):
    dir = GetMayaConfigPath(showDir)
    filename = os.path.join(dir, filename)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data


def GetProtoFile(filename):
    if os.path.splitext(filename)[-1] == '.abc':
        return GetProtoAlembicFile(filename)
    else:
        splitPath = filename.split('/')
        index = splitPath.index('asset')
        task  = splitPath[index + 2]
        if 'element' in splitPath:
            index = splitPath.index('element')
        name = splitPath[index + 1]
        return name, task, filename

def GetProtoAlembicFile(filename):
    showDir, showName = GetProjectPath(maya=filename)
    splitPath = filename.split('/')
    if filename.find(showDir) > -1:
        index = splitPath.index('asset')
        assetName = splitPath[index+1]
        assetDir  = string.join(splitPath[:index+2], '/')
        assetFile = os.path.join(assetDir, assetName + '.usd')
    else:
        assetName = splitPath[-1].split('_model')[0]
        assetFile = '{DIR}/asset/{NAME}/{NAME}.usd'.format(DIR=showDir, NAME=assetName)
    return assetName, 'model', assetFile
