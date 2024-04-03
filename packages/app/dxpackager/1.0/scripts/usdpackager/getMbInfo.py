#coding:utf-8
import maya.standalone
maya.standalone.initialize()

import maya.cmds as cmds
import maya.mel as mel
import os, sys, json

if len(sys.argv) < 3:
    os._exit(1)

mbInfo = {}

mbPath = sys.argv[1]
if not os.path.isfile(mbPath):
    os._exit(1)

if len(sys.argv) > 3 and sys.argv[3] == 'mb2abc':
    cmds.loadPlugin('AbcImport')
    cmds.loadPlugin('AbcExport')

    cmds.file(mbPath, o=True, f=True)
    stF = cmds.playbackOptions(q=True, min=True)
    enF = cmds.playbackOptions(q=True, max=True)
    # meshList = cmds.ls(type='mesh')
    # camList = cmds.ls(type='camera', visible=True)
    # exportList = meshList+camList
    # exportParnetList = []
    # for exp in exportList:
    #     try:
    #         expPrt = cmds.listRelatives(exp, parent=True, type='transform')
    #         if len(expPrt) > 0:
    #             exportParnetList.append(expPrt[0])
    #     except: pass
    exportList = cmds.ls(assemblies=True, visible=True)
    if len(exportList) > 0:
        cmds.select(exportList, r=True)
        mbInfo['startFrame'] = stF
        mbInfo['endFrame'] = enF
        abcPath = os.path.splitext(mbPath)[0]+'.abc'
        print '[ mayapy ] export abc: ', abcPath
        mel.eval('AbcExport -j "-frameRange '+str(int(stF))+' '+str(int(enF))+' -ro -uvWrite -worldSpace -writeVisibility -autoSubd -writeUVSets -dataFormat ogawa -file '+abcPath+'";')

        cmds.file(new=True, f=True)
        cmds.playbackOptions(ast=stF)
        cmds.playbackOptions(aet=enF)
        cmds.playbackOptions(min=stF)
        cmds.playbackOptions(max=enF)
        # cmds.file(abcPath, i=True, type='Alembic', ignoreVersion=True, ra=True, mergeNamespacesOnClash=False, pr=True, importFrameRate=True, importTimeRange='override')

        abcMbPath = os.path.splitext(mbPath)[0]+'-frame.mb'
        cmds.file(rename=abcMbPath)
        print '[ mayapy ] save scene', abcMbPath
        if not abcMbPath.startswith('/show/') and not abcMbPath.startswith('/mach/'):
            cmds.file(s=True, f=True)
            
        if not mbPath.startswith('/show/') and not mbPath.startswith('/mach/'):
            try: os.remove(mbPath)
            except: print '[ mayapy ] delete failed', mbPath

        jsonPath = os.path.splitext(mbPath)[0]+'.json'
        with open(jsonPath, 'w+') as f:
            f.write(json.dumps(mbInfo))

else:
    cmds.file(mbPath, o=True, f=True, loadNoReferences=True)

    referenceList = cmds.file(q=True, r=True)
    if len(referenceList) > 0:
        mbInfo['refList'] = referenceList
        mbInfo['refFileList'] = []
        for r in referenceList:
            print '[ mayapy ] found reference: ', r
            if len(sys.argv) > 3 and sys.argv[3] == 'ref2low':
                rReslv = '/show/'+r.split('/show/')[-1].split('{')[0]
                if rReslv.count('_low.m') < 1:
                    lowFn = rReslv.replace('_mid.m', '.m').replace('_high.m', '.m').replace('.m', '_low.m')
                    refNode = cmds.referenceQuery(r, referenceNode=True)
                    if os.path.isfile(lowFn):
                        cmds.file(lowFn, loadReference=refNode)
                        print '[ mayapy ] reference replace: ', r, '->', lowFn
                        mbInfo['refFileList'].append(lowFn)
                    elif os.path.isfile(rReslv):
                        cmds.file(rReslv, loadReference=refNode)
                        print '[ mayapy ] reference replace: ', r, '->', rReslv
                    else:
                        print '[ mayapy ] reference file not found: ', rReslv

            if r.count('{') < 1:
                mbInfo['refFileList'].append(r)

        if len(sys.argv) > 3 and sys.argv[3] == 'ref2low':
            sn = cmds.file(q=True, sn=True)
            print '[ mayapy ] save scene', sn
            if not sn.startswith('/show/') and not sn.startswith('/mach/'):
                cmds.file(s=True, f=True)
   
jsonPath = sys.argv[2]
with open(jsonPath, 'w+') as f:
    f.write(json.dumps(mbInfo))

os._exit(0)