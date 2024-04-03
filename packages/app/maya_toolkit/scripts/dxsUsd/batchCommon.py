import sys, os, getpass, string
import optparse

try:
    import maya.cmds as cmds
except:
    pass

import dxsMsg


def InitPlugins(plugins):
    unplugins = [
        'bifrostvisplugin', 'bifrostshellnode', 'ZArachneForMaya', 'mtoa',
        'xgenToolkit', 'xgSplineDataToXpd',
        # 'pxrUsd', 'pxrUsdTranslators'     # if maya auto load this plugin, error occurred. so remove this
    ]
    for p in unplugins:
        if cmds.pluginInfo(p, q=True, l=True):
            cmds.unloadPlugin(p)
            dxsMsg.Print('info', 'unload plugin -> %s' % p)
    for p in plugins:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)
        dxsMsg.Print('info', 'plugin -> %s %s' % (p, cmds.pluginInfo(p, q=True, l=True)))

def PathParser(outdir):
    splitPath = outdir.split('/')
    index = splitPath.index('show')
    showDir = string.join(splitPath[:index+2], '/')
    index = splitPath.index('shot')
    seqName = splitPath[index+1]
    shotName= splitPath[index+2]
    return showDir, seqName, shotName

def GetMayaFilename(filename):
    splitExt = os.path.splitext(filename)
    new = splitExt[0].split('--')[0] + splitExt[-1]
    new = new.replace('/CacheOut_Submitter', '')
    return new

def GetZennAssetInfo(showDir):
    infoRule = '{DIR}/_config/maya/AssetInfo.json'
    infoFile = infoRule.format(DIR=showDir)
    if not os.path.exists(infoFile):
        pathRuleFile = '{DIR}/_config/maya/pathRule.json'.format(DIR=showDir)
        if os.path.exists(pathRuleFile):
            with open(pathRuleFile, 'r') as f:
                ruleData = eval(f.read())
                if ruleData.has_key('refShow') and ruleData['refShow']:
                    for sdir in ruleData['refShow']:
                        infoFile = infoRule.format(DIR=sdir)
                        if os.path.exists(infoFile):
                            break
    if not os.path.exists(infoFile):
        print "# Waring : Not found '%s' file" % infoFile
        return None
    with open(infoFile, 'r') as f:
        data = eval(f.read())
        if data.has_key('zenn'):
            return data['zenn']
        else:
            print "# Waring : Not found 'zenn' in '%s' file." % infoFile


def sceneOptParserSetup():
    optparser = optparse.OptionParser()

    optparser.add_option(
        '--mayaver', dest='mayaver', type='string', default='2017',
        help='execute maya scene version'
    )
    optparser.add_option(
        '--srcfile', dest='srcfile', type='string', default='',
        help='export maya filename'
    )
    optparser.add_option(
        '--outdir', dest='outdir', type='string', default='',
        help='output shot directory. ex> /show/ssr_pub/shot/AST/AST_0410'
    )

    optparser.add_option(
        '--all', dest='allexport', action='store_true', default=False,
        help='export all data'
    )
    # export data type
    optparser.add_option(
        '--mesh', dest='mesh', type='string', default='',
        help='export mesh for dxRig.'
             'ex> "v001=tiger:tiger_rigGRP,leech:leech_rig_GRP;v002=tiger1:tiger_rig_GRP"'
    )
    optparser.add_option(
        '--camera', dest='camera', type='string', default='', help='export camera'
    )
    optparser.add_option(
        '--layout', dest='layout', type='string', default='', help='export environment set'
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
    optparser.add_option(
        '--crowdbake', dest='crowdbake', action='store_true', default=False, help='crowd only BakeSkinning'
    )

    optparser.add_option(
        '--zenn', dest='zenn', action='store_true', default=False, help='zenn cache export'
    )
    optparser.add_option(
        '--onlyzenn', dest='onlyzenn', action='store_true', default=False, help='only zenn cache export'
    )
    optparser.add_option(
        '--serial', dest='serial', action='store_true', default=False,
        help='true is one node export'
    )

    optparser.add_option(
        '--rigUpdate', dest='rigUpdate', action='store_true', default=False,
        help='true is rig version latest update'
    )

    # frameRange
    optparser.add_option(
        '--fr', dest='frameRange', type='int', nargs=2, default=(0, 0),
        help='frame range. start end'
    )
    optparser.add_option(
        '--step', dest='step', type='float', default=1.0,
        help='frame step size'
    )

    optparser.add_option(
        '--host', dest='host', type='string', default='local',
        help='compute host. local or tractor or spool. (default: "local")'
    )
    optparser.add_option(
        '--user', dest='user', type='string', default=getpass.getuser(), help='user name'
    )
    return optparser


#-------------------------------------------------------------------------------
#
#   Batch Zenn
#
#-------------------------------------------------------------------------------
def zennOptParserSetup():
    optparser = optparse.OptionParser()
    optparser.add_option(
        '--inputCache', dest='inputCache', type='string', default='',
        help='import geometry cache'
    )
    optparser.add_option(
        '--zennFile', dest='zennFile', type='string', default='',
        help='zenn asset maya file'
    )
    optparser.add_option(
        '--zennNode', dest='zennNode', type='string', default='',
        help='zenn export nodename'
    )
    optparser.add_option(
        '--outdir', dest='outdir', type='string', default='',
        help='output data directory'
    )
    optparser.add_option(
        '--version', dest='version', type='string', default='',
        help='zenn export version'
    )
    # frameRange
    optparser.add_option(
        '--fr', dest='frameRange', type='int', nargs=2, default=(None, None),
        help='frame range. start end'
    )
    optparser.add_option(
        '--step', dest='step', type='float', default=1.0,
        help='frame step size'
    )
    optparser.add_option(
        '--task', dest='task', type='string', default='geom',
        help='zenn export task "geom" or "payload"'
    )
    optparser.add_option(
        '--user', dest='user', type='string', default=getpass.getuser(), help='user name'
    )
    return optparser


#-------------------------------------------------------------------------------
#
#   Batch Crowd
#
#-------------------------------------------------------------------------------
def crowdOptParserSetup():
    optparser = optparse.OptionParser()

    optparser.add_option(
        '--srcfile', dest='srcfile', type='string', default='',
        help='export maya filename'
    )

    optparser.add_option(
        '--outdir', dest='outdir', type='string', default='',
        help='output version directory. ex> /show/ssr_pub/shot/AST/AST_0100/crowd/v001'
    )

    optparser.add_option(
        '--fr', dest='frameRange', type='int', nargs=2, default=(0, 0),
        help='frame range. (start, end)'
    )
    optparser.add_option(
        '--expfr', dest='exportFrameRange', type='int', nargs=2, default=(0, 0),
        help='export frame range. (start, end)'
    )

    optparser.add_option(
        '--task', dest='task', type='string', default='',
        help='export task. [geom or payload or ""]'
    )

    optparser.add_option(
        '--onlybake', dest='onlybake', action='store_true', default=False,
        help='only bake skin'
    )

    optparser.add_option(
        '--meshdrive', dest='meshdrive', action='store_true', default=False,
        help='export meshdrive cache'
    )

    optparser.add_option(
        '--user', dest='user', type='string', default='',
        help='user name'
    )
    return optparser

#-------------------------------------------------------------------------------
#
#   Shot DB Information
#
#-------------------------------------------------------------------------------
def shotDBOptParserSetup():
    optparser = optparse.OptionParser()

    optparser.add_option(
        '--showDir', dest='showDir', type='string', default='',
        help='export show directory root path'
    )

    optparser.add_option(
        '--shot', dest='shot', type='string', default='',
        help='shot name'
    )

    optparser.add_option(
        '--user', dest='user', type='string', default='',
        help='user name'
    )

    optparser.add_option(
        '--name', dest='name', type='string', default='',
        help='nsLayer or assetName'
    )
    optparser.add_option(
        '--version', dest='version', type='string', default='',
        help='cache out version'
    )

    optparser.add_option(
        '--type', dest='type', type='string', default='',
        help='task type ( ani, set, cam, zenn, sim, crowd )'
    )

    optparser.add_option(
        '--outDir', dest='outDir', type='string', default='',
        help='memory calculate directory'
    )

    return optparser
