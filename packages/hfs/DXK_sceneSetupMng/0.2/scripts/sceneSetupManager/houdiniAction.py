# -*- coding: utf-8 -*-
import hou
import os
import pipeCore
import _alembic_hom_extensions as abc
import zenvjson
import DX_Import2

DEBUG = False

def getWorkSpace():
    show = None
    seq = None
    shot = None
    current = hou.hipFile.name()
    if not current:
        return None

    src = current.split('/')
    if 'show' in src:
        show = src[src.index('show') + 1]
    if 'shot' in src:
        seq = src[src.index('shot') + 1]
        shot = src[src.index('shot') + 2]
    return show, seq, shot

currentpath = os.path.dirname(os.path.abspath(__file__))

def SetupEnvironment(mayaversion):
    _env = os.environ.copy()
    _env['CURRENT_LOCATION'] = os.path.dirname(currentpath)
    _env['MAYA_VER'] = str(mayaversion)
    _env['RMAN_VER'] = '21.6'
    _env['BACKSTAGE_PATH'] = '/netapp/backstage/pub'
    _env['BACKSTAGE_MAYA_PATH'] = '%s/apps/maya2' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_RMAN_PATH'] = '%s/apps/renderman2' % _env['BACKSTAGE_PATH']
    _env['BACKSTAGE_ZELOS_PATH'] = '%s/lib/zelos' % _env['BACKSTAGE_PATH']

    _env['RMANTREE'] = '%s/applications/linux/RenderManProServer-%s' % (_env['BACKSTAGE_RMAN_PATH'], _env['RMAN_VER'])
    _env['RMS_SCRIPT_PATHS'] = '%s/rfm-extensions/%s' % (_env['BACKSTAGE_RMAN_PATH'], _env['RMAN_VER'])

    _env['MAYA_LOCATION'] = '/usr/autodesk/maya%s' % mayaversion

    if currentpath.find('/WORK_DATA') > -1:
        _env['BACKSTAGE_MAYA_PATH'] = '/WORK_DATA/script_work/maya2'

    module_path = '%s/modules:%s/modules/%s' % (_env['BACKSTAGE_MAYA_PATH'], _env['BACKSTAGE_RMAN_PATH'], _env['RMAN_VER'])
    _env['MAYA_MODULE_PATH'] = module_path

    _env['LD_LIBRARY_PATH'] = '%s:%s/lib/extern/lib' % (_env['LD_LIBRARY_PATH'], _env['BACKSTAGE_PATH'])
    _env['LD_LIBRARY_PATH'] = '%s:%s/lib' % (_env['LD_LIBRARY_PATH'], _env['BACKSTAGE_ZELOS_PATH'])
    _env['LD_LIBRARY_PATH'] = '%s:%s/maya/%s' % (_env['LD_LIBRARY_PATH'], _env['BACKSTAGE_ZELOS_PATH'], _env['MAYA_VER'])
    _env['LD_LIBRARY_PATH'] = '%s:%s/lib' % (_env['LD_LIBRARY_PATH'], _env['RMANTREE'])

    _env['PATH'] = '%s:%s/lib/extern/bin' % (_env['PATH'], _env['BACKSTAGE_PATH'])

    return _env

class Debug():
    @staticmethod
    def Log(*args):
        if DEBUG:
            log = ""
            for i in args:
                log += str(i)
            print log


# MODEMAP = { 'mesh': 0, 'GPU': 1 }
# WORLDMAP = { 'none': 0, 'baked': 1, 'seperate': 2 }

def script(axis):
    str = """import pipeCore
return pipeCore.camScale({})
""".format(axis)
    return str


def addModeParm(node, file, mode):
    fileNameParm = node.parm("fileName").parmTemplate()

    modeScript = '''
import os
import pipeCore

node = hou.pwd()
file = node.parm("fileName").eval()
mode = node.parm("asbMode").eval()

modeMap = {0: "low", 1: "mid", 2: "high", 3: "sim"}
setFile = pipeCore.getArcFileName( file, modeMap[mode] )
node.parm("fileName").set(setFile)
'''

    parm_grp = node.parmTemplateGroup()

    asbModeParm = hou.MenuParmTemplate("asbMode", "mode", menu_items=(["low", "mid", "high", "sim"]),
                                       menu_labels=(["low", "mid", "high", "sim"]), default_value=0, icon_names=([]),
                                       item_generator_script="",
                                       item_generator_script_language=hou.scriptLanguage.Python,
                                       menu_type=hou.menuType.Normal)
    asbModeParm.setScriptCallback(modeScript)
    asbModeParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    asbModeParm.setTags({"script_callback": modeScript, "script_callback_language": "python"})
    parm_grp.insertBefore(fileNameParm, asbModeParm)
    node.setParmTemplateGroup(parm_grp)


def addXformModeParm(node, file, mode):
    fileNameParm = node.parm("pre_xform").parmTemplate()

    modeScript = '''
import os
import pipeCore

node = hou.pwd()
print node
mode = node.parm("asbMode").eval()

modeMap = {0: "low", 1: "mid", 2: "high", 3: "sim"}

for child in node.children():
    child.parm("asbMode").set(mode)
    child.parm('asbMode').pressButton()
'''

    parm_grp = node.parmTemplateGroup()

    asbModeParm = hou.MenuParmTemplate("asbMode", "mode", menu_items=(["low", "mid", "high", "sim"]),
                                       menu_labels=(["low", "mid", "high", "sim"]), default_value=0, icon_names=([]),
                                       item_generator_script="",
                                       item_generator_script_language=hou.scriptLanguage.Python,
                                       menu_type=hou.menuType.Normal)
    asbModeParm.setScriptCallback(modeScript)
    asbModeParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    asbModeParm.setTags({"script_callback": modeScript, "script_callback_language": "python"})
    parm_grp.insertAfter(fileNameParm, asbModeParm)
    node.setParmTemplateGroup(parm_grp)


def addAbcForXformModeParm(node, file, mode):
    fileNameParm = node.parm("fileName").parmTemplate()

    modeScript = '''
import os
import pipeCore

node = hou.pwd()
file = node.parm("fileName").eval().replace('.wrd', '.abc')
mode = node.parm("asbMode").eval()

modeMap = {0: "low", 1: "mid", 2: "high", 3: "sim"}
setFile = pipeCore.getArcFileName( file, modeMap[mode] )
node.parm("geoFileName").set(setFile)
'''

    parm_grp = node.parmTemplateGroup()

    asbModeParm = hou.MenuParmTemplate("asbMode", "mode", menu_items=(["low", "mid", "high", "sim"]),
                                       menu_labels=(["low", "mid", "high", "sim"]), default_value=0, icon_names=([]),
                                       item_generator_script="",
                                       item_generator_script_language=hou.scriptLanguage.Python,
                                       menu_type=hou.menuType.Normal)
    asbModeParm.setScriptCallback(modeScript)
    asbModeParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    asbModeParm.setTags({"script_callback": modeScript, "script_callback_language": "python"})
    parm_grp.insertBefore(fileNameParm, asbModeParm)

    geoFileParm = hou.StringParmTemplate("geoFileName", "geoFilePath", num_components=1)
    parm_grp.insertBefore(fileNameParm, geoFileParm)
    node.setParmTemplateGroup(parm_grp)


def importCamGeo(cameraGeoPath):
    ### Camera ###
    node = hou.pwd()
    camNode = node.createNode("alembicarchive", "dxCamGeo")
    camNode.parm("fileName").set(cameraGeoPath)
    camNode.parm("loadmode").set(1)
    camNode.parm("buildHierarchy").pressButton()

    return camNode


def importCamLoc(cameraLocPath, nodeName):
    ### Camera ###
    node = hou.pwd()
    nodeName = nodeName.replace(":", "_")
    camNode = node.createNode("alembicarchive", nodeName)
    camNode.parm("fileName").set(cameraLocPath)
    camNode.parm("loadmode").set(1)
    camNode.parm("buildHierarchy").pressButton()

    for node in camNode.allSubChildren():
        if "Shape" in node.name():
            parent = node.parent()
            locNode = parent.createNode("null", parent.name() + "loc")
            node.destroy()
            locNode.setNextInput(parent.indirectInputs()[0])

    return camNode


def importCamera(cameraPath, imageplaneJsonPath = ""):
    fps = hou.fps()

    ### Camera ###
    node = hou.pwd()
    camNode = node.createNode("alembicarchive", "dxCamera")
    camNode.parm("fileName").set(cameraPath)
    camNode.parm("loadmode").set(1)
    camNode.parm("buildHierarchy").pressButton()

    for i in camNode.allSubChildren():
        if (i.type().name() == "cam"):
            i.parmTuple("s").deleteAllKeyframes()
            i.parm("sx").setExpression(script(0), language=hou.exprLanguage.Python)
            i.parm("sy").setExpression(script(1), language=hou.exprLanguage.Python)
            i.parm("sz").setExpression(script(2), language=hou.exprLanguage.Python)

    ### 2D Panzoom ###
    panzoomPath = cameraPath.replace('.abc', '.panzoom')
    if os.path.exists(panzoomPath):
        panzoomJson = pipeCore.readJson(panzoomPath)
        for i in panzoomJson['2DPanZoom'].keys():
            tempNode = camNode.recursiveGlob("*" + i)[0]
            tempNode.parmTuple("win").deleteAllKeyframes()
            tempNode.parmTuple("winsize").deleteAllKeyframes()
            scaleX = 1 / pipeCore.camZoom(tempNode, i, 0)
            scaleY = 1 / pipeCore.camZoom(tempNode, i, 1)
            pipeCore.setKey(panzoomJson['2DPanZoom'][i]['hpn'],
                            tempNode.parm("winx"),
                            fps,
                            (panzoomJson['2DPanZoom'][i]['hpn'].has_key('frame')) and 1 or 0,
                            scaleX)

            pipeCore.setKey(panzoomJson['2DPanZoom'][i]['vpn'],
                            tempNode.parm("winy"),
                            fps,
                            (panzoomJson['2DPanZoom'][i]['vpn'].has_key('frame')) and 1 or 0,
                            scaleY)

            pipeCore.setKey(panzoomJson['2DPanZoom'][i]['zom'],
                            tempNode.parm("winsizex"),
                            fps,
                            (panzoomJson['2DPanZoom'][i]['zom'].has_key('frame')) and 1 or 0)

            pipeCore.setKey(panzoomJson['2DPanZoom'][i]['zom'],
                            tempNode.parm("winsizey"),
                            fps,
                            (panzoomJson['2DPanZoom'][i]['zom'].has_key('frame')) and 1 or 0)

    ### Imageplane ###
    if imageplaneJsonPath  and os.path.exists(imageplaneJsonPath ):
        imageplaneJson = pipeCore.readJson(imageplaneJsonPath )
        for i in imageplaneJson['ImagePlane'].keys():
            tempNode = camNode.recursiveGlob("*" + i)[0]
            ipstr = imageplaneJson["ImagePlane"][i][imageplaneJson["ImagePlane"][i].keys()[0]]["imageName"]["value"]
            ipstr = ipstr.replace(ipstr.split(".")[-2], "$F")
            tempNode.parm("vm_background").set(ipstr)

            res = []
            res.append(imageplaneJson["ImagePlane"][i][imageplaneJson["ImagePlane"][i].keys()[0]]["coverageX"]["value"])
            res.append(imageplaneJson["ImagePlane"][i][imageplaneJson["ImagePlane"][i].keys()[0]]["coverageY"]["value"])
            tempNode.parm("resx").set(res[0])
            tempNode.parm("resy").set(res[1])

    return camNode


def importAssem(assemJsonPath):
    Debug.Log("assemPath : {0}".format(assemJsonPath))
    return zenvjson.import_data(assemJsonPath)
    # # asbMode = hou.pwd().parm("asbAutoMode").eval()
    # asbMode = 2 # 2 : high
    # asbModeString = 'high'
    # tempList = []
    # subnetName = '_Layout'
    #
    # Debug.Log(hou.pwd())
    # # node = hou.pwd().parent()
    # node = hou.node('/obj')
    #
    # geoNodeName = assemPath.split("/")[-1].split(".asb")[0]
    # geoNode = node.createNode("geo", geoNodeName)
    # geoNode.setColor(hou.Color((1, 0, 0)))
    # geoNode.setDisplayFlag(0)
    # geoNode.moveToGoodPosition()
    #
    # hou.node(geoNode.path() + "/file1").destroy()
    #
    # asbNode = geoNode.createNode("alembic", "Import_abs")
    # asbNode.parm("fileName").set(assemPath)
    # asbNode.parm("loadmode").set(2)
    # asbNode.parm("reload").pressButton()
    # asbNode.moveToGoodPosition()
    #
    # rtpNode = geoNode.createNode("attribwrangle", "set_attr")
    # rtpNode.parm("class").set(0)
    # rtpNode.parm("snippet").set('s@varmap = "rtp -> RTP";')
    # rtpNode.setFirstInput(asbNode)
    #
    # geo = asbNode.geometry()
    #
    # arcfiles = geo.findPrimAttrib("arcfiles").strings()
    #
    # switchNode = geoNode.createNode("switch")
    #
    # copyNode = geoNode.createNode("copy")
    # copyNode.parm("pack").set(1)
    # copyNode.parm("stamp").set(1)
    # copyNode.parm("param1").set("rtp")
    # copyNode.parm("val1").setExpression("$RTP")
    #
    # switchNode.parm("input").setExpression('stamp("' + copyNode.path() + '","rtp",0)')
    #
    # copyNode.setFirstInput(switchNode)
    # copyNode.setNextInput(rtpNode)
    #
    # for i in range(len(arcfiles)):
    #     baseName = os.path.splitext(os.path.basename(arcfiles[i]))[0]
    #     importNode = geoNode.createNode('alembic', 'import_%s_%s' % (baseName, i))
    #     fn = pipeCore.getArcFileName(arcfiles[i], asbModeString)
    #     importNode.parm('fileName').set(fn)
    #     importNode.parm('loadmode').set(2)
    #     addModeParm(importNode, arcfiles[i], asbMode)
    #     importNode.parm('asbMode').set(asbMode)
    #     #        importNode.parm('asbMode').pressButton()
    #     switchNode.insertInput(i, importNode)
    #
    # outNode = geoNode.createNode("null", "OUT")
    # outNode.setFirstInput(copyNode)
    # outNode.setDisplayFlag(True)
    # outNode.setRenderFlag(True)
    #
    # for i in geoNode.allSubChildren():
    #     i.moveToGoodPosition()
    # tempList.append(outNode.parent())
    #
    # return geoNode
    #
    # # hou.node("/obj").collapseIntoSubnet(tempList, subnetName)


def importGeo(geoPath, worldOpt=None, alembicOpt=None):
    wrdAutoMode = 2  # hou.pwd().parm("wrdAutoMode").eval()
    wrdAutoModeString = 'high'  # hou.pwd().parm("wrdAutoMode").eval()

    # geoCacheList = []
    # tempList = []
    # for i in geoList:
    #     geoCacheList.append(i.replace(".abc", ".wrd"))
    #     tempList.append(importWRD(i.replace(".abc", ".wrd"), wrdAutoMode)[1])

    # node = hou.pwd().parent()
    node = hou.pwd()

    wrdPath = geoPath.replace('.abc', '.wrd')

    if (os.path.exists(wrdPath)):
        xformNodeName = wrdPath.split("/")[-1].split(".wrd")[0]
        xformNodeName = xformNodeName.replace(":", "_")
        xformNode = node.createNode("alembicxform", xformNodeName)
        time = xformNode.parm("frame").eval() / xformNode.parm("fps").eval()

        hierachyList = []
        typeList = []
        abcList = abc.alembicGetSceneHierarchy(wrdPath, "")[2]

        for i in abcList:
            pipeCore.expandChild("/", i, hierachyList, typeList)

        worldCon = []

        for i in hierachyList:
            if ":world_CON" in i:
                worldCon.append(i)

        xformNode.parm("fileName").set(wrdPath)
        xformNode.parm("objectPath").set(worldCon[0])
        xformNode.parm("frame").setExpression("$FF")
        xformNode.setColor(hou.Color((0, 1, 0)))
        xformNode.setDisplayFlag(1)
        xformNode.moveToGoodPosition()

        geoNode = xformNode.createNode("geo", xformNodeName)
        hou.node(geoNode.path() + "/file1").destroy()

        str = '''import pipeCore
return pipeCore.initScale()

'''
        #        geoNode.parm("scale").setExpression(str,language =hou.exprLanguage.Python)
        print "xformNode :", xformNode.type().name(), xformNodeName
        print "geoNode :", geoNode, geoNode.type().name()
        geoNode.parm("scale").setExpression(str, language=hou.exprLanguage.Python)

        geoNode.setDisplayFlag(1)
        geoNode.moveToGoodPosition()

        connectSubnet(geoNode, 0)

        abcNode = geoNode.createNode("alembic", xformNodeName)

        # wrdMode = hou.parm('/obj/DX_Import21/wrdAutoMode').evalAsString()
        addAbcForXformModeParm(xformNode, wrdPath.replace("wrd", "abc"), wrdAutoModeString)
        abcNode.parm("fileName").set('`chs("../../geoFileName")`')
        xformNode.parm("asbMode").set(wrdAutoModeString)
        xformNode.parm("asbMode").pressButton()
        outNode = geoNode.createNode("null", "OUT")
        outNode.setFirstInput(abcNode)
        outNode.setDisplayFlag(1)
        outNode.setRenderFlag(True)
        outNode.moveToGoodPosition()
        out = [abcNode, abcNode.parent().parent()]
        return out

    #########################################################################################
    else:
        geoNodeName = wrdPath.split("/")[-1].split(".wrd")[0]
        geoNodeName = geoNodeName.replace(":", "_")
        geoNode = node.createNode("geo", geoNodeName)
        hou.node(geoNode.path() + "/file1").destroy()

        geoNode.setDisplayFlag(1)
        geoNode.moveToGoodPosition()

        connectSubnet(geoNode, 0)

        abcNode = geoNode.createNode("alembic", geoNodeName)
        abcNode.parm("fileName").set(wrdPath.replace("wrd", "abc"))

        # wrdMode = hou.parm('/obj/DX_Import21/wrdAutoMode').evalAsString()
        addModeParm(abcNode, wrdPath.replace("wrd", "abc"), wrdAutoModeString)
        abcNode.parm("asbMode").set(wrdAutoModeString)
        abcNode.parm("asbMode").pressButton()
        outNode = geoNode.createNode("null", "OUT")
        outNode.setFirstInput(abcNode)
        outNode.setDisplayFlag(1)
        outNode.setRenderFlag(True)
        outNode.moveToGoodPosition()
        out = [abcNode, abcNode.parent()]
        return out


def connectSubnet(node, inputIndex):
    input = node.parent().indirectInputs()[inputIndex]
    node.setNextInput(input)


def importAsset(assetPath):
    node = hou.pwd()

    wrdAutoModeString = 'high'  # hou.pwd().parm("wrdAutoMode").eval()

    geoNodeName = assetPath.split("/")[-1].split(".abc")[0]

    geoNodeName = geoNodeName.replace(":", "_")
    geoNode = node.createNode("geo", geoNodeName)
    hou.node(geoNode.path() + "/file1").destroy()

    geoNode.setDisplayFlag(1)
    geoNode.moveToGoodPosition()

    abcNode = geoNode.createNode("alembic", geoNodeName)
    abcNode.parm("fileName").set(assetPath)

    # wrdMode = hou.parm('/obj/DX_Import21/wrdAutoMode').evalAsString()
    addModeParm(abcNode, assetPath, wrdAutoModeString)
    abcNode.parm("asbMode").set(wrdAutoModeString)
    abcNode.parm("asbMode").pressButton()
    outNode = geoNode.createNode("null", "OUT")
    outNode.setFirstInput(abcNode)
    outNode.setDisplayFlag(1)
    outNode.setRenderFlag(True)
    outNode.moveToGoodPosition()
    out = [abcNode, abcNode.parent()]
    return out


def importCache(startFrame=None,
                endFrame=None,
                camData={},
                assemData={},
                geoData={},
                zennData={},
                alembicOpt='Mesh',
                worldOpt='None',
                shot=""):
    shotNode = hou.node('/obj').createNode('subnet', '{0}'.format(shot))
    hou.cd(shotNode.path())

    Debug.Log("pwd : {0}".format(hou.pwd().path()))

    dataList = []

    ### CAMERA ###
    camNetBox = None
    if camData:
        Debug.Log(camData)

        camNetBox = hou.pwd().createNetworkBox("cameraNetwotkBox")
        camNetBox.setComment("camera")
        camNetBox.setColor(hou.Color((1, 0.529, 0.624)))

        if camData.has_key('camera_path'):
            for cameraPath in camData['camera_path']:
                if os.path.exists(cameraPath):
                    imageplaneJson = ""
                    if camData.has_key("imageplane_json_path"):
                        imageplaneJson = camData['imageplane_json_path'][0]
                    camNode = importCamera(cameraPath, imageplaneJson)
                    camNetBox.addNode(camNode)

        if camData.has_key('camera_geo_path'):
            for cameraPath in camData['camera_geo_path']:
                if os.path.exists(cameraPath):
                    camNode = importCamGeo(cameraPath)
                    camNetBox.addNode(camNode)

        if camData.has_key('camera_loc_path'):
            for cameraPath in camData['camera_loc_path']:
                if os.path.exists(cameraPath):
                    camNode = importCamLoc(cameraPath, "camLoc")
                    camNetBox.addNode(camNode)
                # dataList.append(camNode)

        if camData.has_key('camera_asset_loc_path'):
            for cameraPath in camData['camera_asset_loc_path']:
                if os.path.exists(cameraPath):
                    camNode = importCamLoc(cameraPath, 'camAssetLoc')
                    camNetBox.addNode(camNode)

        divideCount = 10
        for index, child in enumerate(camNetBox.nodes()):
            child.setPosition(hou.Vector2(((index / divideCount) * 8), (((index % divideCount) + 1))))

    ### import Assem ###
    layNodeBox = None
    if assemData.has_key('json'):

        layNodeBox = hou.pwd().createNetworkBox("layoutNetwotkBox")
        layNodeBox.setComment("layout")
        layNodeBox.setColor(hou.Color((0.29, 0.565, 0.886)))

        for assemPath in assemData['json']:
            if os.path.exists(assemPath):
                # importAssem(assemPath)
                asbNode = importAssem(assemPath)
                Debug.Log("asbNode : {0}".format(str(asbNode)))
                layNodeBox.addNode(asbNode)
                # dataList.append(asbNode)

    ### import GeoCache ###
    geoNodeBox = None
    if geoData:
        subnetName = "geoCache"

        geoNodeBox = hou.pwd().createNetworkBox("geoCacheNetwotkBox")
        geoNodeBox.setComment("geoCache")
        geoNodeBox.setColor(hou.Color((0.765, 1, 0.576)))

        tempList = []
        for asset in geoData.keys():
            if asset in ['maya_files', 'maya_dev_file']:
                continue

            assetPath = geoData[asset]['path'][0]
            if os.path.exists(assetPath):
                tempList.append(importGeo(assetPath, worldOpt, alembicOpt)[1])

        subnet = hou.pwd().collapseIntoSubnet(tempList, subnetName)
        addXformModeParm(subnet, "", "")
        subnet.setDisplayFlag(True)

        divideCount = 10
        input1Pos = subnet.indirectInputs()[0].position()
        for index, child in enumerate(subnet.children()):
            connectSubnet(child, 0)
            child.setPosition(
                hou.Vector2(input1Pos.x() + ((index / divideCount) * 8), input1Pos.y() - (((index % divideCount) + 1))))

        geoNodeBox.addNode(subnet)

    Debug.Log(dataList)
    Debug.Log(shot)

    if len(hou.pwd().iterNetworkBoxes()) > 0:
        nodePosX = 0
        for netNode in hou.pwd().iterNetworkBoxes():
            netNode.fitAroundContents()

        input1Pos = hou.pwd().indirectInputs()[0].position()
        for netNode in hou.pwd().iterNetworkBoxes():
            print hou.Vector2(nodePosX, 0)
            netNode.setPosition(hou.Vector2(input1Pos.x() + nodePosX, input1Pos.y() - 2))
            nodePosX += netNode.size().x() + 0.5
    else:
        print "sceneSetup Fail"

    ### FRAME ###
    fps = hou.fps()
    tset = "tset {0} {1}".format((startFrame - 1) / fps, endFrame / fps)
    hou.hscript(tset)
    hou.playbar.setPlaybackRange(startFrame, endFrame)


def importAssetCache(assetData=None):
    print assetData

    hou.cd(hou.node('/obj').path())

    Debug.Log("pwd : {0}".format(hou.pwd().path()))

    ### MODEL
    if assetData.has_key('model_path'):
        tempList = []
        if os.path.exists(assetData['model_path'][0]):
            subnetName = "assetData"
            #
            # geoNodeBox = hou.pwd().createNetworkBox("geoCacheNetwotkBox")
            # geoNodeBox.setComment("geoCache")
            # geoNodeBox.setColor(hou.Color((0.765, 1, 0.576)))
            #
            # tempList = []
            # for asset in geoData.keys():
            #     assetPath = geoData[asset]['path'][0]
            #     if os.path.exists(assetPath):
            tempList.append(importAsset(assetData['model_path'][0])[1])

            subnet = hou.pwd().collapseIntoSubnet(tempList, subnetName)
            addXformModeParm(subnet, "", "")
            subnet.setDisplayFlag(True)

            # geoNodeBox.addNode(subnet)

    if assetData.has_key('assembly_json'):
        layNodeBox = hou.pwd().createNetworkBox("AssetLayoutNetwotkBox")
        layNodeBox.setComment("layout")
        layNodeBox.setColor(hou.Color((0.29, 0.565, 0.886)))

        for assemPath in assetData['assembly_json']:
            if os.path.exists(assemPath):
                # importAssem(assemPath)
                asbNode = importAssem(assemPath)
                Debug.Log("asbNode : {0}".format(str(asbNode)))
                layNodeBox.addNode(asbNode)

    if assetData.has_key('rig_path'):
        tempList = []
        if os.path.exists(assetData['rig_path'][0]):
            subnetName = "assetData"
            #
            # geoNodeBox = hou.pwd().createNetworkBox("geoCacheNetwotkBox")
            # geoNodeBox.setComment("geoCache")
            # geoNodeBox.setColor(hou.Color((0.765, 1, 0.576)))
            #
            # tempList = []
            # for asset in geoData.keys():
            #     assetPath = geoData[asset]['path'][0]
            #     if os.path.exists(assetPath):
            tempList.append(importAsset(assetData['rig_path'][0])[1])

            subnet = hou.pwd().collapseIntoSubnet(tempList, subnetName)
            addXformModeParm(subnet, "", "")
            subnet.setDisplayFlag(True)

    # if len(hou.pwd().iterNetworkBoxes()) > 0:
    #     nodePosX = 0
    #     for netNode in hou.pwd().iterNetworkBoxes():
    #         netNode.fitAroundContents()
    #
    #     for netNode in hou.pwd().iterNetworkBoxes():
    #         netNode.setPosition(hou.Vector2(nodePosX, 0))
    #         nodePosX += netNode.size().x() + 0.5
    # else:
    #     print "sceneSetup Fail"


def updateCache(startFrame=None, endFrame=None, camData={}, assemData={}, geoData={}, zennData={}, alembicOpt='Mesh',
                worldOpt='None', shot=""):
    node = hou.node('/obj/' + shot)
    if node == None:
        return False

    dataList = []

    ### CAMERA ###
    if camData:
        camNode = node.glob('dxCamera*')
        Debug.Log("camNode : ", camNode)

        camNetBox = node.findNetworkBox('cameraNetworkBox')

        if camNode:
            camExistPath = camNode[0].evalParm("fileName")
            Debug.Log("camExistPath :", camExistPath)
            for cameraPath in camData['camera_path']:
                if os.path.exists(cameraPath) and cameraPath != camExistPath:
                    camNode[0].parm("fileName").set(cameraPath)
                    camNode[0].parm("buildHierarchy").pressButton()
        else:
            for cameraPath in camData['camera_path']:
                if os.path.exists(cameraPath):
                    camNetBox.addNode(importCamera(cameraPath))
                    # dataList.append(importCamera(cameraPath))

    ### import Assem ###
    if assemData:
        assemNode = node.recursiveGlob('*', hou.nodeTypeFilter.ObjGeometry)

        assemNetBox = node.findNetworkBox('layoutNetworkBox')

        Debug.Log("assemNode :", assemNode)
        if assemNode:
            for assem in assemNode:
                assem.destroy()
        for assemPath in assemData['path']:
            if os.path.exists(assemPath):
                # importAssem(assemPath)
                assemNetBox.addNode(importAssem(assemPath))
                # dataList.append(importAssem(assemPath))

    ### import GeoCache ###
    if geoData:
        subnetName = "geoCache"

        # geoNetBox = node.findNetworkBox("geoCacheNetworkBox")

        tempList = []
        for asset in geoData.keys():
            Debug.Log("geoCache Key : ", asset)
            assetPath = geoData[asset]['path'][0]
            assetNode = node.node('geoCache').glob('%s*' % asset)
            Debug.Log("assetPath : ", assetPath)
            Debug.Log("assetNode : ", assetNode)

            if os.path.exists(assetPath):
                if assetNode:
                    assetExistPath = assetNode[0].evalParm('fileName')
                    Debug.Log("assetExistPath :", assetExistPath)
                    if assetPath != assetExistPath:
                        assetNode[0].parm('fileName').set(assetPath)
                else:
                    tempList.append(importGeo(assetPath, worldOpt, alembicOpt)[1])

        hou.moveNodesTo(tempList, node.node('geoCache'))

        divideCount = 5
        for index, child in enumerate(node.node('geoCache').children()):
            connectSubnet(child, 0)
            child.setPosition(hou.Vector2((index / divideCount) * 10, ((index % divideCount) + 1) * 1.5))

    nodePosX = 0
    for netNode in hou.node('/obj/%s' % shot).iterNetworkBoxes():
        netNode.fitAroundContents()

    for netNode in hou.node('/obj/%s' % shot).iterNetworkBoxes():
        netNode.setPosition(hou.Vector2(nodePosX, 0))
        nodePosX += netNode.size().x() + 0.5

    ### FRAME ###
    fps = hou.fps()
    tset = "tset {0} {1}".format((startFrame - 1) / fps, endFrame / fps)
    hou.hscript(tset)
    hou.playbar.setPlaybackRange(startFrame, endFrame)

# def importLight(filePath):
#     lightDic = pipeCore.readJson(filePath)
#     nodeGraph = lightDic["RenderManAsset"]['asset']['nodeGraph']
#     for nodeType in lightDic["RenderManAsset"]['usedNodeTypes']:
#         if nodeType == "PxrDomeLight":
#             lightNode = node.createNode("envlight", nodeType)

def importLight(show, seq, shot):
    node = hou.pwd()
    print show, shot

    outputPath = DX_Import2.run(show, shot)
    DX_Import2.setLight(outputPath)

    # remove Temp JsonFile
    print "rm -rf %s" % outputPath
    # os.system("rm -rf %s" % outputPath)

# def run(show, seq, shot):
#     _env = SetupEnvironment(2017)
#
#     mayaSceneDir = '/show/{show}/shot/{seq}/{shot}/lighting/pub/scenes'.format(show=show, shot=shot, seq=seq)
#
#     if not os.path.exists(mayaSceneDir):
#         mayaSceneDir = mayaSceneDir.replace('/pub/', '/dev/')
#
#     mayaListBeforeSort = []
#     for mayaSceneFile in glob.glob('{0}/*.mb'.format(mayaSceneDir)):
#         mayaListBeforeSort.append(mayaSceneFile)
#
#         mayaListBeforeSort.sort(key=os.path.getmtime)
#         mayaListBeforeSort.reverse()
#
#     if len(mayaListBeforeSort) > 0:
#         mayaScenePath = mayaListBeforeSort[0]
#
#     #    mayaScenePath = '/show/{show}/shot/{seq}/{shot}/lighting/pub/scenes/TTL_1590_lgt_v02_w03.mb'
#
#     #   /show/gcd1/shot/TTL/TTL_1590/ani/pub/data/TTL_1590_ani_v03.json
#     print mayaScenePath
#     sceneFileName = os.path.basename(mayaScenePath)
#     outputDirPath = os.path.dirname(hou.hipFile.path())
#     outputFileName = os.path.splitext(sceneFileName)[0]
#
#     outputPath = "{0}/{1}.json".format(outputDirPath, outputFileName)
#
#     print outputPath
#
#     cmd = '/usr/autodesk/maya2017/bin/mayapy %s/batchGetLightingInfo.py %s %s' % (
#     currentpath, mayaScenePath, outputPath)
#     print cmd
#
#     process = subprocess.Popen(cmd, env=_env, shell=True)
#     process.wait()
#
#     print "END"
#     print "rm -rf", outputPath
#
#     return outputPath
#
#     # setLight(outputPath)
#
# #    os.system()
#
# def setLight(filePath):
#     with open(filePath, 'r') as jsonData:
#         data = json.load(jsonData)
#
#     startFrame = data['frameRange'][0]
#     endFrame = data['frameRange'][1]
#
#     hou.playbar.setPlaybackRange(startFrame, endFrame)
#
#     node = hou.node('/obj')
#
#     lgtNetBox = node.createNetworkBox("lightingNetwotkBox")
#     lgtNetBox.setComment("light_" + os.path.splitext(os.path.basename(filePath))[0])
#     lgtNetBox.setColor(hou.Color((0.996, 0.933, 0)))
#
#     #    nullNode = node.createNode('null', os.path.splitext(os.path.basename(filePath))[0])
#     for frame in range(startFrame, endFrame + 1):
#         hou.setFrame(frame)
#         frame = str(frame)
#         for lightShape in data[frame].keys():
#             nodeType = data[frame][lightShape]['type']
#             lightNode = node.node(lightShape)
#             if lightNode == None:
#                 if nodeType == "PxrDomeLight":
#                     lightNode = node.createNode("envlight", lightShape)
#                     hdriFilePath = data[frame][lightShape]['lightColorMap']
#                     if '.tex' == os.path.splitext(hdriFilePath)[1]:
#                         hdriFilePath = hdriFilePath.replace('.tex', '.hdr')
#                     lightNode.parm("env_map").set(hdriFilePath)
#                 else:
#                     lightNode = node.createNode("hlight::2.0", lightShape)
#                     if nodeType == "PxrDistantLight":
#                         type = 'distant'
#                     else:
#                         type = 'point'
#                     lightNode.parm('light_type').set(type)
#                 lgtNetBox.addNode(lightNode)
#
#             # lightNode.setParmTransform(hou.Matrix4(data[frame][lightShape]['xForm']))i
#             matrix = hou.Matrix4(data[frame][lightShape]['xForm'])
#             t = matrix.explode()['translate']
#             r = matrix.explode()['rotate']
#             s = matrix.explode()['scale']
#             houKeyFrame = hou.Keyframe()
#             houKeyFrame.setTime(hou.frameToTime(int(frame)))
#
#             print lightNode
#             houKeyFrame.setValue(t[0])
#             lightNode.parm('tx').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(t[1])
#             lightNode.parm('ty').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(t[2])
#             lightNode.parm('tz').setKeyframe(houKeyFrame)
#
#             houKeyFrame.setValue(r[0])
#             lightNode.parm('rx').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(r[1])
#             lightNode.parm('ry').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(r[2])
#             lightNode.parm('rz').setKeyframe(houKeyFrame)
#
#             houKeyFrame.setValue(s[0])
#             lightNode.parm('sx').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(s[1])
#             lightNode.parm('sy').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(s[2])
#             lightNode.parm('sz').setKeyframe(houKeyFrame)
#
#             houKeyFrame.setValue(data[frame][lightShape]['intensity'])
#             lightNode.parm('light_intensity').setKeyframe(houKeyFrame)
#
#             houKeyFrame.setValue(data[frame][lightShape]['exposure'])
#             lightNode.parm('light_exposure').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(data[frame][lightShape]['lightColor'][0][0])
#             lightNode.parm('light_colorr').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(data[frame][lightShape]['lightColor'][0][1])
#             lightNode.parm('light_colorg').setKeyframe(houKeyFrame)
#             houKeyFrame.setValue(data[frame][lightShape]['lightColor'][0][2])
#             lightNode.parm('light_colorb').setKeyframe(houKeyFrame)
#
#     gy = 0
#     for child in lgtNetBox.nodes():
#         child.setPosition(hou.Vector2(0, gy))
#         gy += 1
#
#     lgtNetBox.fitAroundContents()
#
#     resize = lgtNetBox.size()
#     resize = resize.__add__(hou.Vector2(3, 0))
#     print resize
#     lgtNetBox.setSize(resize)
