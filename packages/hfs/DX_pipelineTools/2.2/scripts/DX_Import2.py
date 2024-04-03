import hou
import os, sys
import string
import pipeCore
import zenvjson
import subprocess
import json
import glob

currentpath = os.path.dirname(os.path.abspath(__file__))

def script(axis):
    str="""import pipeCore
return pipeCore.camScale({})
""".format(axis)
    return str


def jsonInfo():
    jsonPath = hou.parm("jsonPath").evalAsString()
    if(os.path.exists(jsonPath)):
        js = pipeCore.readJson(jsonPath)
        hou.pwd().setParms({"shot":(js["AlembicCache"].has_key('SHOT')) and js["AlembicCache"]["SHOT"] or "",
            "show": (js["AlembicCache"].has_key('SHOW')) and js["AlembicCache"]["SHOW"] or "",
            "frame":(js["AlembicCache"].has_key('start') and js["AlembicCache"].has_key('end') ) and str(js["AlembicCache"]["start"])+"~"+str(js["AlembicCache"]["end"]) or "",
            "camera":(js["AlembicCache"].has_key('abc_camera')) and js["AlembicCache"]["abc_camera"] or "X",
            "geoCache":(js["AlembicCache"].has_key('mesh')) and "O" or "X",       
            "layout":(js["AlembicCache"].has_key('layout')) and "O" or "X"
            })             
    else:            
        hou.pwd().setParms({"shot":"",
            "show":"",
            "frame":"",
            "camera":"",
            "geoCache":"",       
            "layout":""
            })              
###########################################################################################################
# 0. Import AUTO JSON Button###############################################################################
########################################################################################################### 
def importAutoCamMain():
    try:   
        jsonPath = hou.parm("jsonPath").evalAsString()
        if(os.path.exists(jsonPath)):
            js = pipeCore.readJson(jsonPath)
            if js["AlembicCache"].has_key('abc_camera'):
                importCAM(js["AlembicCache"]['abc_camera'])
            else:
                hou.ui.displayMessage("No cam")
                
    except:
        hou.ui.displayMessage("No cam")



def importAutoWrdMain():
    try:   
        jsonPath = hou.parm("jsonPath").evalAsString()
        if(os.path.exists(jsonPath)):
            js = pipeCore.readJson(jsonPath)
            geoList=[]
            subnetName=""
            if js["AlembicCache"].has_key('mesh'):
                for i in js["AlembicCache"]['mesh']:
                    geoList.append(i)
            if js["AlembicCache"].has_key('SHOT'):
                subnetName = js["AlembicCache"]["SHOT"]+"_GeoCache"
       
        hou.parm("jsonPath").evalAsString()
        wrdAutoMode = hou.pwd().parm("wrdAutoMode").eval()
        geoCacheList=[]
        tempList=[]
        for i in geoList:
            geoCacheList.append(i.replace(".abc",".wrd"))
            tempList.append(importWRD(i.replace(".abc",".wrd"),wrdAutoMode)[1])
        subnet=hou.node("/obj").collapseIntoSubnet(tempList,subnetName)
        subnet.setDisplayFlag(True)

    except:
        hou.ui.displayMessage("importGeoCache ERROR")
        
        
def importAutoLayoutMain():
    try:   
        jsonPath = hou.parm("jsonPath").evalAsString()
        if(os.path.exists(jsonPath)):
            js = pipeCore.readJson(jsonPath)
            geoList=[]
            subnetName=""
            if js["AlembicCache"].has_key('layout'):
                for i in js["AlembicCache"]['layout']:
                    geoList.append(i)
            if js["AlembicCache"].has_key('SHOT'):
                subnetName = js["AlembicCache"]["SHOT"]+"_Layout"
       
        hou.parm("jsonPath").evalAsString()
        asbAutoMode = hou.pwd().parm("asbAutoMode").eval()
        geoCacheList=[]
        tempList=[]
        for i in geoList:
            geoCacheList.append(i)
            tempList.append(importLayout(i,asbAutoMode).parent())
        hou.node("/obj").collapseIntoSubnet(tempList,subnetName)
        
    except:
        hou.ui.displayMessage("importGeoCache ERROR")        
        
###########################################################################################################
# 0. Import JSON Button##################################################################################
###########################################################################################################        
def setFrame():
    try:    
        jsonPath = hou.parm("jsonPath").evalAsString()
        js = pipeCore.readJson(jsonPath)
        start= js["AlembicCache"]["start"]
        end= js["AlembicCache"]["end"]
        jsonPath = hou.parm("jsonPath").evalAsString()
        js = pipeCore.readJson(jsonPath)

#setFPS        
        if(js["AlembicCache"]["fps"]=="film"):
            hou.setFps(24)
        elif(js["AlembicCache"]["fps"]=="ntsc"):
            hou.setFps(30)        

        fps = hou.fps()
        msg="Change Frame Range..?\n shotDB Range : %s - %s" % (int(start), int(end))
        if hou.ui.displayMessage(msg, buttons=("Yes", "No"), severity=hou.severityType.ImportantMessage) == 0:
            set = 'tset {0} {1}'.format((start-1)/fps,end/fps)
            hou.hscript(set)
            hou.playbar.setPlaybackRange(start,end)    
    except:
        hou.ui.displayMessage("importJSON ERROR")            


###########################################################################################################
# 1. Import Camera Button##################################################################################
###########################################################################################################        
def importCAM(camPath):
    try:
        if(os.path.exists(camPath)):
            panzoomPath = camPath.replace(".abc",".panzoom")
            imageplanePath = camPath.replace(".abc",".imageplane")   
            fps = hou.fps()      
    
            node = hou.node("/obj/")
            camNode = node.createNode("alembicarchive", "cam")
            camNode.parm("fileName").set(camPath)
            camNode.parm("loadmode").set(1)
            camNode.parm("buildHierarchy").pressButton()
    
            for i in camNode.allSubChildren():
                if(i.type().name()=="cam"):
                    i.parmTuple("s").deleteAllKeyframes()
                    i.parm("sx").setExpression(script(0),language =hou.exprLanguage.Python)
                    i.parm("sy").setExpression(script(1),language =hou.exprLanguage.Python)
                    i.parm("sz").setExpression(script(2),language =hou.exprLanguage.Python)
                 
            #2d PanZoom
            if(os.path.exists(panzoomPath)):
                pzjs = pipeCore.readJson(panzoomPath)
                for i in pzjs["2DPanZoom"].keys():
                    tempNode = camNode.recursiveGlob("*"+i)[0]
                    tempNode.parmTuple("win").deleteAllKeyframes()
                    tempNode.parmTuple("winsize").deleteAllKeyframes()
                    scaleX = 1/pipeCore.camZoom(tempNode,i,0)
                    scaleY = 1/pipeCore.camZoom(tempNode,i,1)
                    pipeCore.setKey(pzjs["2DPanZoom"][i]["hpn"],tempNode.parm("winx"),fps,(pzjs["2DPanZoom"][i]["hpn"].has_key('frame')) and 1 or 0,scaleX)
                    pipeCore.setKey(pzjs["2DPanZoom"][i]["vpn"],tempNode.parm("winy"),fps,(pzjs["2DPanZoom"][i]["vpn"].has_key('frame')) and 1 or 0,scaleY)
                    pipeCore.setKey(pzjs["2DPanZoom"][i]["zom"],tempNode.parm("winsizex"),fps,(pzjs["2DPanZoom"][i]["zom"].has_key('frame')) and 1 or 0)
                    pipeCore.setKey(pzjs["2DPanZoom"][i]["zom"],tempNode.parm("winsizey"),fps,(pzjs["2DPanZoom"][i]["zom"].has_key('frame')) and 1 or 0)

            #Imageplane set                
            if(os.path.exists(imageplanePath)):
                ipjs = pipeCore.readJson(imageplanePath)
                for i in ipjs["ImagePlane"].keys():
                    tempNode = camNode.recursiveGlob("*"+i)[0]
                    ipstr = ipjs["ImagePlane"][i][ipjs["ImagePlane"][i].keys()[0]]["imageName"]["value"]
                    ipstr = ipstr.replace(ipstr.split(".")[-2],"$F")
                    tempNode.parm("vm_background").set(ipstr)
    
                    res=[]
                    res.append(ipjs["ImagePlane"][i][ipjs["ImagePlane"][i].keys()[0]]["coverageX"]["value"])
                    res.append(ipjs["ImagePlane"][i][ipjs["ImagePlane"][i].keys()[0]]["coverageY"]["value"])
                    tempNode.parm("resx").set(res[0])
                    tempNode.parm("resy").set(res[1])

        else:
            hou.ui.displayMessage("No cam") 

    except:
        hou.ui.displayMessage("importCAM ERROR")

def importJsonLayout(jsonPath):
    zenvjson.import_data(jsonPath)
        
###########################################################################################################
# 2. Import Layout Button##################################################################################
###########################################################################################################      

def importLayout(asbPath,asbMode):
    node = hou.pwd().parent()
    
    geoNodeName= asbPath.split("/")[-1].split(".asb")[0]
    geoNode = node.createNode("geo",geoNodeName)
    geoNode.setColor(hou.Color((1,0,0)))
    geoNode.setDisplayFlag(0)
    geoNode.moveToGoodPosition()
    
    hou.node(geoNode.path()+"/file1").destroy()
    
    asbNode = geoNode.createNode("alembic", "Import_abs")
    asbNode.parm("fileName").set(asbPath)
    asbNode.parm("loadmode").set(2)
    asbNode.parm("reload").pressButton()
    asbNode.moveToGoodPosition()
    
    
    rtpNode = geoNode.createNode("attribwrangle","set_attr")
    rtpNode.parm("class").set(0)
    rtpNode.parm("snippet").set('s@varmap = "rtp -> RTP";')
    rtpNode.setFirstInput(asbNode)
    
    geo = asbNode.geometry()

    arcfiles = geo.findPrimAttrib("arcfiles").strings()

    switchNode = geoNode.createNode("switch")
    
    copyNode = geoNode.createNode("copy")
    copyNode.parm("pack").set(1)
    copyNode.parm("stamp").set(1)
    copyNode.parm("param1").set("rtp")
    copyNode.parm("val1").setExpression("$RTP")

    switchNode.parm("input").setExpression('stamp("'+copyNode.path() +'","rtp",0)')
    
    
    copyNode.setFirstInput(switchNode)
    copyNode.setNextInput(rtpNode)


    for i in range(len(arcfiles)):
        baseName   = os.path.splitext( os.path.basename(arcfiles[i]) )[0]
        importNode = geoNode.createNode( 'alembic', 'import_%s_%s' % (baseName, i) )
        fn = pipeCore.getArcFileName( arcfiles[i], asbMode.evalAsString() )
        importNode.parm('fileName').set( fn )
        importNode.parm('loadmode').set( 2 )
        addModeParm( importNode, arcfiles[i], asbMode )
        importNode.parm('asbMode').set( asbMode )
#        importNode.parm('asbMode').pressButton()
        switchNode.insertInput( i, importNode )

    outNode = geoNode.createNode("null", "OUT")
    outNode.setFirstInput(copyNode)
    outNode.setDisplayFlag(True)
    outNode.setRenderFlag(True)

    for i in geoNode.allSubChildren():
        i.moveToGoodPosition()
    return outNode        

    
    
###########################################################################################################
# 3. Import WRD  Button  ##################################################################################
###########################################################################################################      

def importWRD(wrdPath,wrdMode):
    import _alembic_hom_extensions as abc
    node = hou.pwd().parent()
    if(os.path.exists(wrdPath)):
        xformNodeName= wrdPath.split("/")[-1].split(".wrd")[0]
        xformNode = node.createNode("alembicxform",xformNodeName)
        time = xformNode.parm("frame").eval()/xformNode.parm("fps").eval()
    
        hierachyList=[]
        typeList=[]    
        abcList = abc.alembicGetSceneHierarchy(wrdPath,"")[2]
    
        for i in abcList:
            pipeCore.expandChild("/",i,hierachyList,typeList)
    
        worldCon=[]
    
        for i in hierachyList:
            if ":world_CON" in i:
                worldCon.append(i)
                
        xformNode.parm("fileName").set(wrdPath)
        xformNode.parm("objectPath").set(worldCon[0])   
        xformNode.parm("frame").setExpression("$FF")
        xformNode.setColor(hou.Color((0,1,0)))    
        xformNode.setDisplayFlag(1)
        xformNode.moveToGoodPosition()
        
        geoNode = xformNode.createNode("geo",xformNodeName)
        hou.node(geoNode.path()+"/file1").destroy()
        
        str='''import pipeCore
return pipeCore.initScale()
    
'''
#        geoNode.parm("scale").setExpression(str,language =hou.exprLanguage.Python)
        geoNode.parm("scale").setExpression(str,language =hou.exprLanguage.Python)

        geoNode.setDisplayFlag(1)
        geoNode.moveToGoodPosition()
        
    
        abcNode = geoNode.createNode("alembic",xformNodeName)
        abcNode.parm("fileName").set(wrdPath.replace("wrd","abc"))
        
        wrdMode = hou.parm('/obj/DX_Import21/wrdAutoMode').evalAsString()
        addModeParm(abcNode,wrdPath.replace("wrd","abc"),wrdMode)
        abcNode.parm("asbMode").set(wrdMode)
        abcNode.parm("asbMode").pressButton()
        outNode=geoNode.createNode("null","OUT")
        outNode.setFirstInput(abcNode)
        outNode.setDisplayFlag(1)
        outNode.setRenderFlag(True)
        outNode.moveToGoodPosition()
        out=[abcNode,abcNode.parent().parent()]
        return out
        
#########################################################################################        
    else:
        geoNodeName= wrdPath.split("/")[-1].split(".wrd")[0]
 
        geoNode = node.createNode("geo",geoNodeName)                
        hou.node(geoNode.path()+"/file1").destroy()

        geoNode.setDisplayFlag(1)
        geoNode.moveToGoodPosition()

        abcNode = geoNode.createNode("alembic",geoNodeName)
        abcNode.parm("fileName").set(wrdPath.replace("wrd","abc"))
        
        wrdMode = hou.parm('/obj/DX_Import21/wrdAutoMode').evalAsString()
        addModeParm(abcNode,wrdPath.replace("wrd","abc"),wrdMode)
        abcNode.parm("asbMode").set(wrdMode)
        abcNode.parm("asbMode").pressButton()
        outNode=geoNode.createNode("null","OUT")
        outNode.setFirstInput(abcNode)
        outNode.setDisplayFlag(1)
        outNode.setRenderFlag(True)
        outNode.moveToGoodPosition()
        out=[abcNode,abcNode.parent()]
        return out

        
#################################################################################################################          
#################################################################################################################          
#################################################################################################################  
def addModeParm(node,file,mode):
    fileNameParm = node.parm("fileName").parmTemplate()
    
    modeScript='''
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
    
    asbModeParm = hou.MenuParmTemplate("asbMode", "mode",  menu_items=(["low","mid","high","sim"]), menu_labels=(["low","mid","high","sim"]), default_value=0, icon_names=([]), item_generator_script="", item_generator_script_language=hou.scriptLanguage.Python, menu_type=hou.menuType.Normal)
    asbModeParm.setScriptCallback(modeScript)
    asbModeParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    asbModeParm.setTags({"script_callback": modeScript, "script_callback_language": "python"})
    parm_grp.insertBefore(fileNameParm,asbModeParm)
    node.setParmTemplateGroup(parm_grp)

def SetupEnvironment(mayaversion):
    _env = os.environ.copy()
    _env['CURRENT_LOCATION'] = os.path.dirname(currentpath)
    _env['MAYA_VER'] = str(mayaversion)
    _env['RMAN_VER'] = '21.4'
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

def importAutoLight():
    node = hou.pwd()
    print node

    show = node.parm("show").eval()
    shot = node.parm("shot").eval()

    print show, shot

    outputPath = run(show, shot)
    setLight(outputPath)

    # remove Temp JsonFile
    print "rm -rf %s" % outputPath
    # os.system("rm -rf %s" % outputPath)

def run(show, shot):
    _env = SetupEnvironment(2017)

    seq = shot.split('_')[0]

    mayaSceneDir = '/show/{show}/shot/{seq}/{shot}/lighting/pub/scenes'.format(show = show, shot = shot, seq = seq)

    if not os.path.exists(mayaSceneDir):
        mayaSceneDir = mayaSceneDir.replace('/pub/', '/dev/')

    mayaListBeforeSort = []
    for mayaSceneFile in glob.glob('{0}/*.mb'.format(mayaSceneDir)):
        mayaListBeforeSort.append(mayaSceneFile)

        mayaListBeforeSort.sort( key = os.path.getmtime )
        mayaListBeforeSort.reverse()

    if len(mayaListBeforeSort) > 0:
        mayaScenePath = mayaListBeforeSort[0]

#    mayaScenePath = '/show/{show}/shot/{seq}/{shot}/lighting/pub/scenes/TTL_1590_lgt_v02_w03.mb'

#   /show/gcd1/shot/TTL/TTL_1590/ani/pub/data/TTL_1590_ani_v03.json
    print mayaScenePath
    sceneFileName = os.path.basename(mayaScenePath)
    outputDirPath = os.path.dirname(hou.hipFile.path())
    outputFileName = os.path.splitext(sceneFileName)[0]

    outputPath = "{0}/{1}.json".format(outputDirPath, outputFileName)

    print outputPath

    cmd = '/usr/autodesk/maya2017/bin/mayapy %s/batchGetLightingInfo.py %s %s' % (currentpath, mayaScenePath, outputPath)
    print cmd

    process = subprocess.Popen(cmd, env = _env, shell = True)
    process.wait()

    print "END"
    print "rm -rf", outputPath
    
    return outputPath

    # setLight(outputPath)
#    os.system()

def setLight(filePath):
    with open(filePath, 'r') as jsonData:
        data = json.load(jsonData)

    node = hou.node('/obj')

    lgtNetBox = node.createNetworkBox("lightingNetwotkBox")
    lgtNetBox.setComment("light_" + os.path.splitext(os.path.basename(filePath))[0])
    lgtNetBox.setColor(hou.Color((0.996, 0.933, 0)))

#    nullNode = node.createNode('null', os.path.splitext(os.path.basename(filePath))[0])

    for lightShape in data.keys():
        nodeType = data[lightShape]['type']
        if nodeType == "PxrDomeLight":
            lightNode = node.createNode("envlight", lightShape)
            hdriFilePath = data[lightShape]['lightColorMap']
            if '.tex' == os.path.splitext(hdriFilePath)[1]:
                hdriFilePath = hdriFilePath.replace('.tex', '.hdr')
            lightNode.parm("env_map").set(hdriFilePath)
        else:
            lightNode = node.createNode("hlight::2.0", lightShape)
            if nodeType == "PxrDistantLight":
                type = 'distant'
            else:
                type = 'point'
            lightNode.parm('light_type').set(type)
        
        lightNode.setParmTransform(hou.Matrix4(data[lightShape]['xForm']))
        lightNode.parm('light_intensity').set(data[lightShape]['intensity'])
        lightNode.parm('light_exposure').set(data[lightShape]['exposure'])
        lightNode.parm('light_colorr').set(data[lightShape]['lightColor'][0][0])
        lightNode.parm('light_colorg').set(data[lightShape]['lightColor'][0][1])
        lightNode.parm('light_colorb').set(data[lightShape]['lightColor'][0][2])
    
        lgtNetBox.addNode(lightNode)

    gy = 0
    for child in lgtNetBox.nodes():
        child.setPosition(hou.Vector2(0, gy))
        gy += 1
    
    lgtNetBox.fitAroundContents()

    resize = lgtNetBox.size()    
    resize = resize.__add__(hou.Vector2(3, 0))
    print resize
    lgtNetBox.setSize(resize)

