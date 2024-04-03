# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#
#   Jungmin Lee
#
#	2018.04.02
#   2018.05.18 - add renderman setup
#-------------------------------------------------------------------------------

import os, sys
import json
currentpath = os.path.abspath(__file__)

# init environment
def __runtime_environment():
    for p in os.getenv('PYTHONPATH').split(':'):
        if p and not p in sys.path:
            sys.path.append(p)

def SetupEnvironmentUSD(mayaversion):
    _env = os.environ.copy()
    _env['CURRENT_LOCATION'] = os.path.dirname(currentpath)
    _env['MAYA_VER'] = str(mayaversion)
    _env['MAYA_LOCATION'] = '/usr/autodesk/maya%s' % mayaversion
    _env['BACKSTAGE_PATH'] = '/netapp/backstage/pub'
    _env['BACKSTAGE_MAYA_PATH']  = '%s/apps/maya2' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_RMAN_PATH']  = '%s/apps/renderman2' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_ZELOS_PATH'] = '%s/lib/zelos' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_BORA_PATH']  = '%s/lib/bora' % _env['BACKSTAGE_PATH']

    # RenderMan
    # _env['RMANTREE'] = '/opt/pixar/RenderManProServer-21.7'
    # _env['RMS_SCRIPT_PATHS'] = '%s/rfm-extensions/21.7' % _env['BACKSTAGE_RMAN_PATH']

    _env['MAYA_MODULE_PATH'] = _env['BACKSTAGE_MAYA_PATH'] + '/modules'
    # _env['MAYA_MODULE_PATH']+= ':%s/modules/21.7' % _env['BACKSTAGE_RMAN_PATH']

    # USD
    _env['USD_INSTALL_ROOT']  = _env['BACKSTAGE_PATH'] + '/lib/extern/usd/0.8.5'
    _env['MAYA_PLUG_IN_PATH'] = _env['USD_INSTALL_ROOT'] + '/third_party/maya/plugin'
    _env['MAYA_SCRIPT_PATH']  = _env['USD_INSTALL_ROOT'] + '/third_party/maya/share/usd/plugins/usdMaya/resources'
    _env['PYTHONPATH'] = _env['PYTHONPATH'] + ':' + _env['USD_INSTALL_ROOT'] + '/lib/python'
    _env['PYTHONPATH'] = _env['PYTHONPATH'] + ':' + _env['BACKSTAGE_PATH'] + '/lib/extern/lib/python'
    _env['PYTHONPATH'] = _env['PYTHONPATH'] + ':' + _env['BACKSTAGE_PATH'] + '/lib/python_lib'
    _env['XBMLANGPATH']=_env['USD_INSTALL_ROOT'] + '/third_party/maya/share/usd/plugins/usdMaya/resources'

    _env['LD_LIBRARY_PATH'] = '/lib64:/usr/lib64'
    _env['LD_LIBRARY_PATH'] = _env['LD_LIBRARY_PATH'] + ':' + _env['BACKSTAGE_PATH']+'/lib/extern/lib'
    _env['LD_LIBRARY_PATH'] = _env['LD_LIBRARY_PATH'] + ':' + _env['BACKSTAGE_PATH'] + '/lib/extern/cuda/lib64'
    _env['LD_LIBRARY_PATH'] = _env['LD_LIBRARY_PATH'] + ':' + _env['USD_INSTALL_ROOT'] + '/third_party/maya/lib'
    _env['LD_LIBRARY_PATH'] = _env['LD_LIBRARY_PATH'] + ':' + _env['USD_INSTALL_ROOT'] + '/lib'
    _env['LD_LIBRARY_PATH'] = _env['LD_LIBRARY_PATH'] + ':' + _env['BACKSTAGE_ZELOS_PATH'] + '/lib'
    _env['LD_LIBRARY_PATH'] = _env['LD_LIBRARY_PATH'] + ':' + _env['BACKSTAGE_ZELOS_PATH'] + '/maya/%s/python' % mayaversion

    _env['PATH'] = _env['PATH'] + ':' + _env['BACKSTAGE_PATH'] + '/lib/extern/bin'
    _env['PATH'] = _env['PATH'] + ':' + _env['USD_INSTALL_ROOT']+'/bin'
    _env['PATH'] = _env['PATH'] + ':' + '/usr/autodesk/maya%s/bin' % mayaversion

    return _env


def GetShowDir(show):
    _showDir= '/show/%s' % show
    showDir = _showDir
    if showDir.find('_pub') == -1:
        showDir += '_pub'

    pathRuleFile = '{SHOW}/_config/maya/pathRule.json'.format(SHOW=_showDir)
    if os.path.exists(pathRuleFile):
        showDir = _showDir
        try:
            ruleData = json.load(open(pathRuleFile))
            if ruleData.has_key('showDir') and ruleData['showDir']:
                __showDir = ruleData['showDir']
                if os.path.isabs(__showDir):
                    showDir = __showDir
                else:
                    showDir = os.path.join('/show', __showDir)
        except:
            pass
    return showDir

def optParserSetup():
    import optparse
    import getpass

    optparser = optparse.OptionParser()
    optparser.add_option(
        '--srcfile', dest='srcfile', type='string', default='',
        help='export maya filename'
    )
    optparser.add_option(
        '--outdir', dest='outdir', type='string', default='',
        help='output shot directory. ex.>/show/ssr_pub/shot/AST/AST_0410'
    )

    optparser.add_option(
        '--all', dest='allexport', action='store_true', default=False,
        help='all data export'
    )
    # export type
    optparser.add_option(
        '--mesh', dest='mesh', type='string', default='',
        help='export mesh for dxRig.'
             'ex.> "v001=tiger:tiger_rig_GRP,leech:leech_rig_GRP;v002=tirger1:tiger_rig_GRP"'
    )
    optparser.add_option(
        '--camera', dest='camera', type='string', default='', help='export camera'
    )
    optparser.add_option(
        '--layout', dest='layout', type='string', default='', help='export layout'
    )
    optparser.add_option(
        '--sim', dest='simmesh', type='string', default='', help='export simulation'
    )
    optparser.add_option(
        '--hairSim', dest='hairSim', type='string', default='', help='export Zenn of already simulation geoCache'
    )
    optparser.add_option(
        '--crowd', dest='crowd', type='string', default='', help='export crowd scene'
    )

    # frame range
    optparser.add_option(
        '--fr', dest='frame', type='int', nargs=2, default=(0, 0),
        help='frame range. start end')
    optparser.add_option(
        '--stepSize', dest='stepSize', type='float', default=0.0,
        help='frame step size')
    optparser.add_option(
        '--zenn', dest='zenn', action='store_true', default=False,
        help='zenn cache export'
    )
    optparser.add_option(
        '--onlyzenn', dest='onlyzenn', action='store_true', default=False,
        help='only zenn cache export'
    )
    optparser.add_option(
        '--error', dest='error', action='store_true', default=False,
        help='error check'
    )
    optparser.add_option(
        '--rigUpdate', dest='rigUpdate', action='store_true', default=False,
        help='true is rig version lastest update'
    )
    optparser.add_option(
        '--crdBake', dest='crdBake', action='store_true', default=False,
        help='true is crwod data re bake'
    )
    optparser.add_option(
        '--serial', dest='serial', action='store_true', default=False,
        help='true is one node export'
    )

    # local or tractor
    optparser.add_option(
        '--host', dest='host', type='string', default="local",
        help='compute host. local, tractor. (default: "local")'
    )
    optparser.add_option(
        '--user', dest='user', type='string', default=getpass.getuser(),
        help='user name')

    return optparser

def GetZennAssetInfo(showDir):
    filename = '{SHOWDIR}/_config/maya/AssetInfo.json'.format(SHOWDIR=showDir)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = eval(f.read())
            if data.has_key('zenn'):
                return data['zenn']
    else:
        print '# Debug : using DB'

        from pymongo import MongoClient
        import dxConfig
        client = MongoClient(dxConfig.getConf("DB_IP"))
        db = client["PIPE_PUB"]
        showName = showDir.split("/show/")[-1]
        showName = showName.split("_")[0]
        coll = db[showName]

        nameList = coll.find({"task":"asset", "data_type":"zenn", "enabled":True}).distinct("asset_name")
        if nameList:
            data = {}
            for assetName in nameList:
                item = coll.find_one({'task':'asset',
                                       "data_type":"zenn",
                                       "enabled":True,
                                       "asset_name":assetName},
                                      sort = [("version", -1)])
                zennPath = item['files']['scene'][0]
                data[assetName] = {'filename' : zennPath}
            return data
        else:
            print "# Warning : Not found zenn template in DB"
            return None

def getRigVersion(inputCache, task = 'sim'):
    sys.path.append("/netapp/backstage/pub/lib/extern/usd/18.11/lib/python")
    from pxr import Usd
    stage = Usd.Stage.Open(inputCache, load=Usd.Stage.LoadNone)
    dprim = stage.GetDefaultPrim()
    customData = dprim.GetCustomData()
    rigVersion = None
    aniCacheFile = inputCache

    if task == "sim":
        if customData.has_key('simInputCache'):
            aniCacheFile = customData['simInputCache']

    stage = Usd.Stage.Open(aniCacheFile, load=Usd.Stage.LoadNone)
    dprim = stage.GetDefaultPrim()
    customData = dprim.GetCustomData()
    if customData.has_key('rig'):
        rigVersion = customData['rig']
    return rigVersion

def GetLayerVersion(dirPath):
    if os.path.exists(dirPath):
        source = list()
        for i in os.listdir(dirPath):
            if os.path.isdir(os.path.join(dirPath, i)):
                if i[0] == 'v':
                    source.append(i)
        source.sort()
        if source:
            version = 'v%03d' % (int(source[-1][1:]) + 1)
        else:
            version = 'v001'
    else:
        version = 'v001'
    return version