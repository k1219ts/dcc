# Copyright (C) 2018 Dexter Studio VFX Pine-line. All Rights Reserved.

from Katana import FarmAPI, NodegraphAPI, Nodes3DAPI, Callbacks

import os
import sys
import string
import subprocess
import pprint

#-------------------------------------------------------------------------------
#
#   SETUP
#
#-------------------------------------------------------------------------------
def onStartup(**kwargs):
    # Add menu options
    FarmAPI.AddFarmPopupMenuOption('Tractor : Render Spool', RenderProc)
    FarmAPI.AddFarmPopupMenuOption('Tractor : Only Denoise Spool', DenoiseProc)
    FarmAPI.AddFarmPopupMenuOption('LocalQueue Spool', LocalQueueProc)

    # Specified Frames
    FarmAPI.AddFarmSettingString(
        'Specified.frames', hints={'label': 'Frames', 'help': '1-24,30,32,40-45'}
    )
    FarmAPI.AddFarmSettingNumber(
        'Specified.byframe', 1, hints={'label': 'By frame', 'int': True, 'constant': True}
    )

    # Tractor Options
    FarmAPI.AddFarmSettingString(
        'Tractor.engine',
        hints={'label': 'Engine', 'help': 'Applied by last connected node.'}
    )
    FarmAPI.AddFarmSettingString(
        'Tractor.tier',
        hints={'label': 'Tier', 'help': 'Applied by last connected node.'}
    )
    FarmAPI.AddFarmSettingNumber(
        'Tractor.priority', 100,
        hints={'label': 'Priority', 'int': True, 'constant': True,
               'help': 'Applied by last connected node.'}
    )
    FarmAPI.AddFarmSettingString(
        'Tractor.projects',
        hints={'label': 'Projects', 'help': 'Applied by last connected node.'}
    )
    FarmAPI.AddFarmSettingNumber(
        'Tractor.maxactive', 0,
        hints={'label': 'Max Active', 'int': True, 'constant': True,
               'help': 'Applied by last connected node.'}
    )
    FarmAPI.AddFarmSettingString(
        'Tractor.tags',
        hints={'label': 'Limit Tags', 'help': 'Applied by last connected node.'}
    )
    FarmAPI.AddFarmSettingString(
        'Tractor.slots', hints={'label': 'Min/Max Slots', 'help': '2 or 1/4 or 4/4'}
    )

Callbacks.addCallback(Callbacks.Type.onStartup, onStartup)

#-------------------------------------------------------------------------------
def RenderProc():
    node = FarmAPI.GetCurrentNode()
    RenderMainProc(node)

def DenoiseProc():
    sys.stdout.write('\033[1;31m')
    print '[INFO TractorSpool] : Only Denoise'
    sys.stdout.write('\033[0;0m')
    node = FarmAPI.GetCurrentNode()
    RenderMainProc(node, onlyDenoise=True)

def LocalQueueProc():
    # process check
    _process = False
    for i in os.popen('ps -A').readlines():
        if i.find('LocalQueue') > -1 and i.find('defunct') == -1:
            _process = True

    from Katana import UI4
    if not _process:
        UI4.Widgets.MessageBox.Critical("LocalQueue Error", "You need to run 'LocalQueue' first!")
        return

    node = FarmAPI.GetCurrentNode()
    katfile, jobscript = RenderMainProc(node, localQueue=True)
    jobfile = katfile.replace('.katana', '.alf')
    f = open(jobfile, 'w')
    f.write(jobscript)
    f.close()
    args = []
    args.append('%s/bin/LocalQueue' % (os.environ['RMANTREE']))
    args.append(jobfile)
    subprocess.Popen(args)



def InitRender(renderNode):
    '''
    - node : selected Render or RenderScript Node
    '''
    result = dict()
    msgLog = dict()
    for rs in FarmAPI.GetSortedDependencies(renderNode):
        node = NodegraphAPI.GetNode(rs.nodeName)
        if node.getType() == 'Render':
            # SyncOutputPorts
            Nodes3DAPI.RenderNodeUtil.SyncOutputPorts(node)
            if rs.outputs:
                for output in rs.outputs:
                    name = output['name']   # primary, lgt, lpe, geo, ...
                    location = output['outputLocation']
                    if location.find('/tmp/') > -1:
                        msgLog['%s.%s' % (rs.nodeName, name)] = location
                    else:
                        if not result.has_key(rs.nodeName):
                            result[rs.nodeName] = list()
                        result[rs.nodeName].append(os.path.dirname(location))
    return result, msgLog


def RenderMainProc(renderNode, onlyDenoise=False, localQueue=False):
    from Katana import KatanaFile, UI4
    import JobScript.prmanRender as prmanRender
    import JobScript.renderLog as renderLog
    from TractorSpool.spoolMain import SpoolMain

    if not os.getenv('DEV_LOCATION'):
        retMsg = SpoolMain(renderNode=renderNode, parent=UI4.App.MainWindow.GetMainWindow()).exec_()
        if retMsg == 0:
            return

    katfile = FarmAPI.GetKatanaFileName()
    frange  = FarmAPI.GetSceneFrameRange()
    if not katfile:
        UI4.Widgets.MessageBox.Critical('File Error', 'Save the file first!')
        return

    if not renderNode:
        UI4.Widgets.MessageBox.Critical('Render Node Selection Error', '\tBe sure to select only one node!')
        return

    nodeData, msgLog = InitRender(renderNode)
    if msgLog:
        sys.stdout.write('\033[1;31m')
        print '[ERROR OutputLocation] :'
        pprint.pprint(msgLog)
        sys.stdout.write('\033[0;0m')
        UI4.Widgets.MessageBox.Critical('Output Location Error', '\tCheck the renderLocation !\t')
        return

    makeMovParm = renderNode.getParameter('user.makeMov')
    isMakeMov = False
    if makeMovParm:
        isMakeMov = bool(makeMovParm.getValue(0))

    # create output folder
    for n in nodeData:
        for loc in nodeData[n]:
            if not os.path.exists(loc):
                os.makedirs(loc)

    # SAVE
    KatanaFile.Save(katfile)

    # version log
    if not onlyDenoise:
        for n in nodeData:
            renderLog.VersionLog(n, outdir=nodeData[n][0]).doIt()

    renderSettings = FarmAPI.GetSortedDependencies(renderNode)

    sys.stdout.write('\033[1;34m')
    print '[INFO TractorSpool KatanaFile] :', katfile
    print '[INFO Scene FrameRange] :', frange
    print '[INFO Render Nodes] :', nodeData.keys()
    render = prmanRender.JobMain(katfile, frange, renderSettings, onlyDenoise=onlyDenoise, localQueue=localQueue, makeMov=isMakeMov)
    render.titleSuffix = string.join(nodeData.keys(), ',')
    jobscript = render.doIt()
    sys.stdout.write('\033[0;0m')
    return katfile, jobscript
