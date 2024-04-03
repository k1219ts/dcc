#
#
# 3DE4.script.name:	Export Project to Maya for MMV...
#
# 3DE4.script.version:	v1.10.3
#
# 3DE4.script.gui:	Main Window::DD_MMV
#
# 3DE4.script.comment:	Creates a MEL script file that contains all project data, which can be imported into Autodesk Maya.
#
#

# based on export_maya.py(v1.9) script from 3dequalizer4 r3b1
# modified by Dongho Cha after Daehwan Jang

# change log.
# v1.9.4: add some code for sourcing exported mel script in maya.
# v1.9.5: add overscan widget
# v1.9.6: add overscan toggle function
# v1.10.1: ...
# v1.10.2: change Project name
# v1.10.3: add obj traslate, rotate, scale attribute export (kwantae.kim)
# v1.11.0: DEXTER DIGITAL pipeline hierarchy apply (kwantae.kim)

#
# import sdv's python vector lib...
import sys, os
import datetime
import getpass
instpath = tde4.get3DEInstallPath()
sys.path.append("%s/sys_data/py_vl_sdv"%instpath)
from vl_sdv import *
import TDE4_common
reload(TDE4_common)

#
# functions...

def write_log(mel):
    d = datetime.datetime.today()
    today = d.strftime("%y%m%d")
    path = "/dexter/Cache_DATA/MMV/tmp/mel_log/log_%s.txt" % today

    if not path:
        f = open(path, "w")
    else:
        f = open(path, "a")
    f.write("%s %s %s %s %s\n" % (d.strftime("%H:%M:%S"), getpass.getuser(), os.environ["show"], os.environ["shot"], mel))
    f.close()


def convertToAngles(r3d):
    rot	= rot3d(mat3d(r3d)).angles(VL_APPLY_ZXY)
    rx	= (rot[0]*180.0)/3.141592654
    ry	= (rot[1]*180.0)/3.141592654
    rz	= (rot[2]*180.0)/3.141592654
    return(rx,ry,rz)


def convertZup(p3d,yup):
    if yup==1:
        return(p3d)
    else:
        return([p3d[0],-p3d[2],p3d[1]])


def angleMod360(d0,d):
    dd	= d-d0
    if dd>180.0:
        d	= angleMod360(d0,d-360.0)
    else:
        if dd<-180.0:
            d	= angleMod360(d0,d+360.0)
    return d


def validName(name):
    name	= name.replace(" ","_")
    name	= name.replace("-","_")
    name	= name.replace("\n","")
    name	= name.replace("\r","")
    return name

def prepareImagePath(path,startframe):
    path	= path.replace("\\","/")
    i	= 0
    n	= 0
    i0	= -1
    while(i<len(path)):
        if path[i]=='#': n += 1
        if n==1: i0 = i
        i	+= 1
    if i0!=-1:
        fstring		= "%%s%%0%dd%%s"%(n)
        path2		= fstring%(path[0:i0],startframe,path[i0+n:len(path)])
        path		= path2
    return path

def is_overscan_callback(requester, widget, action):
    mode = tde4.getWidgetValue(req, "overscan")
    if mode == 0:
        tde4.setWidgetSensitiveFlag(req, "overscan_v", 0)
    if mode == 1:
        tde4.setWidgetSensitiveFlag(req, "overscan_v", 1)

def overscan_value_validator(requester, widget, action):
    overscan_index = tde4.getWidgetValue(req, "overscan_v")
    #print overscan_index
    if overscan_list[overscan_index-1] == "custom":
        tde4.setWidgetSensitiveFlag(req, "overscan_custom", 1)
    else:
        tde4.setWidgetSensitiveFlag(req, "overscan_custom", 0)

#
# main script...
USER = os.getenv("USER")

window_title = "Export Project To Maya for MMV..."
project_path = tde4.getProjectPath()
file_name = ""
if project_path != None:
    file_name = project_path.replace(".3de",".mel")

j = {}
if os.environ.has_key('show'):
    try:
        for envKey in ['show', 'seq','shot','platetype']:
            j[envKey] = os.environ[envKey]
    except:
        raise Exception("Parsing ENVKEY failed! It goes something wrong.")
else:	#pmodel case
    for envKey in ['show', 'seq','shot','platetype']:
        j[envKey] = ""

#if not j["shot"] in project_path:
#    tde4.postQuestionRequester(window_title,"Open 3DEqualizer Project using *Open Script*!","Ok")

#
# search for camera point group...
campg = None
pgl	= tde4.getPGroupList()
for pg in pgl:
    if tde4.getPGroupType(pg)=="CAMERA":	campg = pg
if campg==None:
    tde4.postQuestionRequester(window_title,"Error, There is no camera point group.","Ok")

cl = tde4.getCameraList(1)
if len(cl)==0:
    tde4.postQuestionRequester(window_title,"Error, Select cameras to export first.","Ok")


#
# open requester...
#try:
req = tde4.createCustomRequester()
tde4.addFileWidget(req,"file_browser","Exportfile...","*.mel", file_name)
tde4.addTextFieldWidget(req, "start_frame", "Start Frame", str(tde4.getCameraSequenceAttr(cl[0])[0]))
tde4.addToggleWidget(req,"stereo","Stereo Camera", 0)

overscan_list = ["1.08", "1.1", "1.15", "1.2", "custom"]

tde4.addToggleWidget(req,"overscan","Overscan?", 0)
tde4.setWidgetCallbackFunction(req, "overscan", "is_overscan_callback")
tde4.addOptionMenuWidget(req, "overscan_v", "Overscan Value", *overscan_list)
tde4.setWidgetCallbackFunction(req, "overscan_v", "overscan_value_validator")
tde4.setWidgetSensitiveFlag(req, "overscan_v", 0)
tde4.addTextFieldWidget(req, "overscan_custom", "Overscan Custom", '')
tde4.setWidgetSensitiveFlag(req, "overscan_custom", 0)

#tde4.setWidgetSensitiveFlag(nuke_node_req, "right_camera", 0)
tde4.addSeparatorWidget(req, "sep01")
tde4.addToggleWidget(req,"export_3dmodel","Export 3D Model", 1)
tde4.addToggleWidget(req,"hide_ref_frames","Hide Reference Frames", 0)

for c in tde4.getCameraList():
    stereo_status = tde4.getCameraStereoMode(c)
if not stereo_status == "STEREO_OFF":
    tde4.setWidgetValue(req,"stereo", "1")

bbox = TDE4_common.bbdld_compute_bounding_box()

overscan_scale = round(bbox[2] / bbox[4], 2)
if overscan_scale < round(bbox[3] / bbox[5], 2):
    overscan_scale = round(bbox[3] / bbox[5], 2)

#print bbox[0], bbox[1], bbox[2], bbox[3], overscan_scale

if bbox[0] < 0.0000 or bbox[1] < 0.0000:
    tde4.setWidgetValue(req, 'overscan', '1')
    tde4.setWidgetSensitiveFlag(req, "overscan_v", 1)

    if overscan_scale > 1.2:
        OVERSCAN = overscan_scale
        tde4.setWidgetValue(req, 'overscan_v', "5")  # custom
        tde4.setWidgetValue(req, 'overscan_custom', str(OVERSCAN))
        tde4.setWidgetSensitiveFlag(req, 'overscan_custom', 1)

    else:
        for i in reversed(overscan_list):
            if i != "custom" and float(i) > overscan_scale:
                OVERSCAN = float(i)
                widgetIdx = overscan_list.index(i) + 1
                tde4.setWidgetValue(req, 'overscan_v', str(widgetIdx))

                #print OVERSCAN, overscan_scale, widgetIdx
else:
    OVERSCAN = 1.00

ret = tde4.postCustomRequester(req, window_title, 600, 0, "Ok","Cancel")
if ret==1:
    # yup	= tde4.getWidgetValue(req,"mode_menu")
    # if yup==2: yup = 0
    yup	= 1
    path	= tde4.getWidgetValue(req,"file_browser")
    frame0	= int(tde4.getWidgetValue(req,"start_frame"))
    frame0	-= 1
    stereo = tde4.getWidgetValue(req, "stereo")

    overscan_v = tde4.getWidgetValue(req, "overscan")
    if overscan_v == 0:
        overscan = 1.0
    else:
        # CHECK IF OVERSCAN IS PRESET OR CUSTOM VALUE
        overscan_index = tde4.getWidgetValue(req, "overscan_v")
        if overscan_list[overscan_index-1] == "custom":
            overscan = float(tde4.getWidgetValue(req, 'overscan_custom'))
        else:
            overscan = float(overscan_list[overscan_index-1])
    #print "OVERSCAN VALUE : ", overscan

    # overscan_v = tde4.getWidgetValue(req, "overscan")
    # if overscan_v == 0:
    #     overscan = 1.0
    # else:
    #     overscan = 1.08
    hide_ref= tde4.getWidgetValue(req,"hide_ref_frames")
    export_model = tde4.getWidgetValue(req,"export_3dmodel")
    if path!=None:
        if not path.endswith('.mel'): path = path+'.mel'
        f	= open(path,"w")
        if not f.closed:

            #
            # write some comments...

            f.write("//\n")
            f.write("// Maya/MEL export data written by %s\n"%tde4.get3DEVersion())
            f.write("// Based on export_maya.py(v1.9) script from 3dequalizer4 r3b1\n")
            f.write("// Modified by Kwantae.Kim, 2019.09.06\n")

            f.write("//\n")
            f.write("// All lengths are in centimeter, all angles are in degree.\n")
            f.write("//\n\n")

            #
            # write dxCameraNode for DEXTER DIGITAL...
            f.write("string $sceneGroupName = `createNode dxCamera`;\n")

            #
            # write cameras...

            #cl = tde4.getCameraList()
            index = 1
            for cam in cl:
                camType = tde4.getCameraType(cam)
                noframes = tde4.getCameraNoFrames(cam)
                lens = tde4.getCameraLens(cam)
                if lens!=None:
                    name = validName(tde4.getCameraName(cam))
                    name = name.split("_")
                    name = "%s_matchmove" % "_".join(name[:-1])

                    index += 1
                    fback_w = tde4.getLensFBackWidth(lens)
                    fback_h = tde4.getLensFBackHeight(lens)
                    p_aspect = tde4.getLensPixelAspect(lens)
                    focal = tde4.getCameraFocalLength(cam,1)
                    lco_x = tde4.getLensLensCenterX(lens)
                    lco_y = tde4.getLensLensCenterY(lens)

                    # convert filmback to inch...
                    fback_w = fback_w/2.54
                    fback_h = fback_h/2.54
                    lco_x = -lco_x/2.54
                    lco_y = -lco_y/2.54

                    # convert focal length to mm...
                    focal = focal*10.0

                    # set render global.
                    image_w = tde4.getCameraImageWidth(cam)
                    image_h = tde4.getCameraImageHeight(cam)
                    f.write("setAttr \"defaultResolution.width\" %s;\n"%image_w)
                    f.write("setAttr \"defaultResolution.height\" %s;\n"%image_h)
                    f.write("setAttr \"defaultResolution.deviceAspectRatio\" %.8f;\n"%(float(image_w)/float(image_h)))
                    f.write("setAttr \"defaultRenderGlobals.animation\" 1;\n")
                    f.write("setAttr \"defaultRenderGlobals.extensionPadding\" 4;\n")
                    f.write("setAttr \"defaultRenderGlobals.putFrameBeforeExt\" 1;\n")
                    f.write("setAttr \"defaultRenderGlobals.startFrame\" %d;\n"%(1+frame0))
                    f.write("setAttr \"defaultRenderGlobals.endFrame\" %d;\n"%(noframes+frame0))

                    # create camera...
                    f.write("\n")
                    f.write("// create camera %s...\n"%name)
                    f.write("createNode \"camera\" - n \"%sShape\";\n" % name)
                    f.write("string $cameraNodes[] = `ls \"%s*\"`;\n" % name)
                    f.write("camera -e -hfa %.15f  -vfa %.15f -fl %.15f -ncp 0.1 -fcp 100000 -shutterAngle 180 -ff \"horizontal\" $cameraNodes[0];\n"%(fback_w*overscan, fback_h*overscan, focal))
                    f.write("string $cameraTransform = $cameraNodes[0];\n")
                    f.write("string $cameraShape = $cameraNodes[1];\n")
                    f.write("xform -zeroTransformPivots -rotateOrder zxy $cameraTransform;\n")
                    f.write("setAttr ($cameraShape+\".horizontalFilmOffset\") %.15f;\n"%lco_x);
                    f.write("setAttr ($cameraShape+\".verticalFilmOffset\") %.15f;\n"%lco_y);
                    p3d = tde4.getPGroupPosition3D(campg,cam,1)
                    p3d = convertZup(p3d,yup)
                    f.write("xform -translation %.15f %.15f %.15f $cameraTransform;\n"%(p3d[0],p3d[1],p3d[2]))
                    r3d = tde4.getPGroupRotation3D(campg,cam,1)
                    rot = convertToAngles(r3d)
                    f.write("xform -rotation %.15f %.15f %.15f $cameraTransform;\n"%rot)
                    f.write("xform -scale 1 1 1 $cameraTransform;\n")

                    # image plane...
                    f.write("\n")
                    f.write("// create image plane...\n")
                    # this occurs fatal error when export camera.
                    # f.write("string $imagePlane = `createNode transform -n \"imagePlane\" -p $cameraShape`;\n")
                    # f.write("createNode imagePlane -n \"imagePlaneShape\" -p $imagePlane;\n")
                    f.write("string $imagePlane = `createNode imagePlane`;\n")
                    f.write("cameraImagePlaneUpdate ($cameraShape, $imagePlane);\n")
                    f.write("setAttr ($imagePlane + \".offsetX\") %.15f;\n"%lco_x)
                    f.write("setAttr ($imagePlane + \".offsetY\") %.15f;\n"%lco_y)

                    if camType=="SEQUENCE": f.write("setAttr ($imagePlane+\".useFrameExtension\") 1;\n")
                    else: f.write("setAttr ($imagePlane+\".useFrameExtension\") 0;\n")

                    f.write("expression -n \"frame_ext_expression\" -s ($imagePlane+\".frameExtension=frame\");\n")

                    tde4.setCameraProxyFootage(cam, 3)
                    if tde4.getCameraPath(cam) == "":
                        tde4.setCameraProxyFootage(cam, 0)
                    path = tde4.getCameraPath(cam)
                    tde4.setCameraProxyFootage(cam, 0)
                    sattr = tde4.getCameraSequenceAttr(cam)
                    path = prepareImagePath(path,sattr[0])
                    f.write("setAttr ($imagePlane + \".imageName\") -type \"string\" \"%s\";\n"%(path))
                    f.write("setAttr ($imagePlane + \".fit\") 4;\n")
                    f.write("setAttr ($imagePlane + \".displayOnlyIfCurrent\") 1;\n")
                    f.write("setAttr ($imagePlane  + \".depth\") (9000/2);\n")

                    # parent camera to scene group...
                    f.write("\n")
                    f.write("// parent camera to scene group...\n")
                    f.write("parent $cameraTransform $sceneGroupName;\n")

                    if camType=="REF_FRAME" and hide_ref:
                        f.write("setAttr ($cameraTransform +\".visibility\") 0;\n")

                    # animate camera...
                    if camType!="REF_FRAME":
                        f.write("\n")
                        f.write("// animating camera %s...\n"%name)
                        f.write("playbackOptions -ast %d -aet %d -min %d -max %d;\n"%(1+frame0, noframes+frame0, 1+frame0, noframes+frame0))
                        f.write("currentTime %d;\n"%(1+frame0))
                        f.write("\n")

                    frame = 1
                    while frame<=noframes:
                        # rot/pos...
                        p3d = tde4.getPGroupPosition3D(campg,cam,frame)
                        p3d = convertZup(p3d,yup)
                        r3d = tde4.getPGroupRotation3D(campg,cam,frame)
                        rot = convertToAngles(r3d)
                        if frame>1:
                            rot = [ angleMod360(rot0[0],rot[0]), angleMod360(rot0[1],rot[1]), angleMod360(rot0[2],rot[2]) ]
                        rot0 = rot
                        f.write("setKeyframe -at translateX -t %d -v %.15f $cameraTransform; "%(frame+frame0,p3d[0]))
                        f.write("setKeyframe -at translateY -t %d -v %.15f $cameraTransform; "%(frame+frame0,p3d[1]))
                        f.write("setKeyframe -at translateZ -t %d -v %.15f $cameraTransform; "%(frame+frame0,p3d[2]))
                        f.write("setKeyframe -at rotateX -t %d -v %.15f $cameraTransform; "%(frame+frame0,rot[0]))
                        f.write("setKeyframe -at rotateY -t %d -v %.15f $cameraTransform; "%(frame+frame0,rot[1]))
                        f.write("setKeyframe -at rotateZ -t %d -v %.15f $cameraTransform; "%(frame+frame0,rot[2]))

                        # focal length...
                        focal = tde4.getCameraFocalLength(cam,frame)
                        focal = focal*10.0
                        f.write("setKeyframe -at focalLength -t %d -v %.15f $cameraShape;\n"%(frame+frame0,focal))

                        frame += 1

            #
            # write scene info...
            f.write("\n")
            f.write("// write scene info...\n")
            f.write("fileInfo \"3deProject\" \"%s\";\n"%project_path)
            if overscan_v:
                f.write("fileInfo \"overscan\" \"true\";\n")
            else:
                f.write("fileInfo \"overscan\" \"false\";\n")
            f.write("fileInfo \"overscan_value\" \"%s\";\n" % str(overscan))

            f.write("fileInfo \"resWidth\" \"%s\";\n"%image_w)
            f.write("fileInfo \"resHeight\" \"%s\";\n"%image_h)
            f.write("fileInfo \"plateType\" \"%s\";\n"%j["platetype"])
            f.write("fileInfo \"show\" \"%s\";\n"%j["show"])
            f.write("fileInfo \"seq\" \"%s\";\n"%j["seq"])
            f.write("fileInfo \"shot\" \"%s\";\n"%j["shot"])
            if stereo:
                f.write("fileInfo \"stereo\" \"true\";\n")
            else:
                f.write("fileInfo \"stereo\" \"false\";\n")
            f.write("fileInfo \"user\" \"%s\";\n"%USER)

            #
            # write camera point group...

            f.write("\n")
            f.write("// create camera point group...\n")
            #name = "cameraPGroup_%s_1"%validName(tde4.getPGroupName(campg))
            name = "cam_loc"
            f.write("string $pointGroupName = `group -em -name  \"%s\" -parent $sceneGroupName`;\n"%name)
            f.write("$pointGroupName = ($sceneGroupName + \"|\" + $pointGroupName);\n")
            f.write("\n")

            # write points...
            l = tde4.getPointList(campg)
            for p in l:
                if tde4.isPointCalculated3D(campg,p):
                    name = tde4.getPointName(campg,p)
                    name = "cam_%s"%validName(name)
                    p3d = tde4.getPointCalcPosition3D(campg,p)
                    p3d = convertZup(p3d,yup)

                    f.write("\n")
                    f.write("// create point %s...\n"%name)
                    f.write("string $locator = stringArrayToString(`spaceLocator -name %s`, \"\");\n"%name)
                    f.write("$locator = (\"|\" + $locator);\n")
                    f.write("xform -t %.15f %.15f %.15f $locator;\n"%(p3d[0],p3d[1],p3d[2]))
                    f.write("parent $locator $pointGroupName;\n")

            f.write("\n")
            f.write("xform -zeroTransformPivots -rotateOrder zxy -scale 1.000000 1.000000 1.000000 $pointGroupName;\n")

            f.write("string $camGeoGroupName = `group -em -name \"cam_geo\"`;\n")
            f.write("parent $camGeoGroupName $sceneGroupName;\n")

            chk_model_name = []

            if export_model == 1:
                for i in tde4.get3DModelList(campg):
                    cnt = 0
                    model_path = tde4.get3DModelFilepath(campg, i)
                    model_name = tde4.get3DModelName(campg, i)
                    model_name = os.path.splitext(model_name)[0]
                    model_name = validName(model_name)

                    if chk_model_name.count(model_name) > 0:
                        cnt = chk_model_name.count(model_name)

                    pos = tde4.get3DModelPosition3D(campg, i, cl[0], 1)
                    r3d = tde4.get3DModelRotationScale3D(campg, i)
                    #print r3d

                    abc1 = mat3d(r3d)
                    abc2 = rot3d(mat3d(r3d)).angles(VL_APPLY_ZXY)

                    #print abc1
                    #print type(abc1)
                    #print abc2
                    #print type(abc2)


                    rot = convertToAngles(r3d)
                    #print rot

                    m_out = [[r3d[0][0], r3d[0][1], r3d[0][2], 0],
                             [r3d[1][0], r3d[1][1], r3d[1][2], 0],
                             [r3d[2][0], r3d[2][1], r3d[2][2], 0],
                             [pos[0], pos[1], pos[2], 1]]

                    if os.path.isfile(model_path):
                        f.write("file -reference -type \"OBJ\" -loadReferenceDepth \"all\" -mergeNamespacesOnClash true -options \"mo=1\" \"%s\";\n"%(model_path))
                        f.write("$referenceOBJ = `ls -tr \"%s*\"`;\n" % (model_name))
                        f.write("xform -roo zxy -matrix %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f $referenceOBJ[%s];\n" %(
                            m_out[0][0], m_out[0][1], m_out[0][2], m_out[0][3],
                            m_out[1][0], m_out[1][1], m_out[1][2], m_out[1][3],
                            m_out[2][0], m_out[2][1], m_out[2][2], m_out[2][3],
                            m_out[3][0], m_out[3][1], m_out[3][2], m_out[3][3], str(cnt)))

                        f.write("xform -ro %f %f %f $referenceOBJ[%s];\n\n" % (rot[0], rot[1], rot[2], str(cnt)))
                        if not model_name.count("spheregrid") > 0:
                            f.write("parent $referenceOBJ[%s] $camGeoGroupName;\n" % str(cnt))
                        #f.write("//setAttr %s_Mesh.s %f %f %f;\n" % (model_name))
                        #f.write("file -import -type \"OBJ\" -ra true -mergeNamespacesOnClash false -rpr \"%s\" -options \"mo=1\" \"%s\";\n"%(model_name, model_path))

                    chk_model_name.append(model_name)

            f.write("\n")

            #
            # write object/mocap point groups...

            camera = tde4.getCurrentCamera()
            noframes = tde4.getCameraNoFrames(camera)
            pgl = tde4.getPGroupList()
            index = 1
            locNum = 1
            for pg in pgl:
                if tde4.getPGroupType(pg)=="OBJECT" and camera!=None:
                    f.write("\n")
                    f.write("// create object point group...\n")
                    if tde4.get3DModelList(pg):
                        model_name = tde4.get3DModelName(pg, tde4.get3DModelList(pg)[0])
                        pgname = "%s_loc" % (model_name.split("_")[0])
                    else:
                        model_name = None
                        pgname = "obj%d_loc" % (index)

                    index += 1
                    f.write("string $pointGroupName = `group -em -name  \"%s\" -parent $sceneGroupName`;\n"%pgname)
                    f.write("$pointGroupName = ($sceneGroupName + \"|\" + $pointGroupName);\n")

                    # write points...
                    l = tde4.getPointList(pg)
                    for p in l:
                        if tde4.isPointCalculated3D(pg,p):
                            name = tde4.getPointName(pg,p)
                            name = "obj%d_%s"%(index-1, validName(name))
                            p3d = tde4.getPointCalcPosition3D(pg,p)
                            p3d = convertZup(p3d,yup)
                            f.write("\n")
                            f.write("// create point %s...\n"%name)
                            f.write("string $locator%d = stringArrayToString(`spaceLocator -name %s`, \"\");\n"%(locNum, name))
                            f.write("$locator%d = (\"|\" + $locator%d);\n"%(locNum,locNum))
                            f.write("xform -t %.15f %.15f %.15f $locator%d;\n"%(p3d[0],p3d[1],p3d[2], locNum))
                            f.write("parent $locator%d $pointGroupName;\n"%locNum)
                            locNum += 1



                    f.write("\n")
                    scale = tde4.getPGroupScale3D(pg)
                    f.write("xform -zeroTransformPivots -rotateOrder zxy -scale %.15f %.15f %.15f $pointGroupName;\n"%(scale,scale,scale))

                    # animate object point group...
                    f.write("\n")
                    f.write("// animating point group %s...\n"%pgname)
                    frame = 1
                    model_keyframe = ""
                    while frame<=noframes:
                        # rot/pos...
                        p3d = tde4.getPGroupPosition3D(pg,camera,frame)
                        p3d = convertZup(p3d,yup)
                        r3d = tde4.getPGroupRotation3D(pg,camera,frame)
                        rot = convertToAngles(r3d)
                        if frame>1:
                            rot = [ angleMod360(rot0[0],rot[0]), angleMod360(rot0[1],rot[1]), angleMod360(rot0[2],rot[2]) ]
                        rot0 = rot
                        f.write("setKeyframe -at translateX -t %d -v %.15f $pointGroupName; "%(frame+frame0,p3d[0]))
                        f.write("setKeyframe -at translateY -t %d -v %.15f $pointGroupName; "%(frame+frame0,p3d[1]))
                        f.write("setKeyframe -at translateZ -t %d -v %.15f $pointGroupName; "%(frame+frame0,p3d[2]))
                        f.write("setKeyframe -at rotateX -t %d -v %.15f $pointGroupName; "%(frame+frame0,rot[0]))
                        f.write("setKeyframe -at rotateY -t %d -v %.15f $pointGroupName; "%(frame+frame0,rot[1]))
                        f.write("setKeyframe -at rotateZ -t %d -v %.15f $pointGroupName;\n"%(frame+frame0,rot[2]))

                        frame += 1

                    if export_model == 1 and model_name:
                        f.write("string $objGeoGroupName = `group -em -name \"%s_geo\"`;\n" % (model_name.split("_")[0]))
                        # f.write("xform -rotateOrder zxy $objGeoGroupName;\n")
                        for i in tde4.get3DModelList(pg):
                            model_path = tde4.get3DModelFilepath(pg, i)
                            model_name = tde4.get3DModelName(pg, i)
                            model_name = os.path.splitext(model_name)[0]
                            model_name = validName(model_name)

                            if os.path.isfile(model_path):
                                f.write("file -reference -type \"OBJ\" -loadReferenceDepth \"all\" -mergeNamespacesOnClash true -options \"mo=1\" \"%s\";\n"%(model_path))
                                f.write("$referenceOBJ = `ls -tr \"|%s*\"`;\n" % (model_name))
                                f.write("xform -rotateOrder zxy $referenceOBJ;\n")
                                f.write("parentConstraint $pointGroupName $referenceOBJ;\n")
                                f.write("parent $referenceOBJ $objGeoGroupName;\n")
                                #f.write(model_keyframe)
                                #f.write("file -import -type \"OBJ\" -ra true -mergeNamespacesOnClash false -rpr \"%s\" -options \"mo=1\" \"%s\";\n"%(model_name, model_path))
                        f.write("parent $objGeoGroupName $sceneGroupName;\n")

                # mocap point groups...
                if tde4.getPGroupType(pg)=="MOCAP" and camera!=None:
                    f.write("\n")
                    f.write("// create mocap point group...\n")
                    pgname = "mocap%d_loc" % index
                    index += 1
                    f.write("string $pointGroupName = `group -em -name  \"%s\" -parent $sceneGroupName`;\n"%pgname)
                    f.write("$pointGroupName = ($sceneGroupName + \"|\" + $pointGroupName);\n")

                    # write points...
                    l = tde4.getPointList(pg)
                    for p in l:
                        if tde4.isPointCalculated3D(pg,p):
                            name = tde4.getPointName(pg,p)
                            name = "mocap%d_%s"%(index-1, validName(name))
                            p3d = tde4.getPointMoCapCalcPosition3D(pg,p,camera,1)
                            p3d = convertZup(p3d,yup)
                            f.write("\n")
                            f.write("// create point %s...\n"%name)
                            f.write("string $locator = stringArrayToString(`spaceLocator -name %s`, \"\");\n"%name)
                            f.write("$locator = (\"|\" + $locator);\n")
                            f.write("xform -t %.15f %.15f %.15f $locator;\n"%(p3d[0],p3d[1],p3d[2]))
                            for frame in range(1,noframes+1):
                                p3d = tde4.getPointMoCapCalcPosition3D(pg,p,camera,frame)
                                p3d = convertZup(p3d,yup)
                                f.write("setKeyframe -at translateX -t %d -v %.15f $locator; "%(frame+frame0,p3d[0]))
                                f.write("setKeyframe -at translateY -t %d -v %.15f $locator; "%(frame+frame0,p3d[1]))
                                f.write("setKeyframe -at translateZ -t %d -v %.15f $locator; "%(frame+frame0,p3d[2]))
                            f.write("parent $locator $pointGroupName;\n")

                    f.write("\n")
                    scale = tde4.getPGroupScale3D(pg)
                    f.write("xform -zeroTransformPivots -rotateOrder zxy -scale %.15f %.15f %.15f $pointGroupName;\n"%(scale,scale,scale))

                    # animate mocap point group...
                    f.write("\n")
                    f.write("// animating point group %s...\n"%pgname)
                    frame = 1
                    while frame<=noframes:
                        # rot/pos...
                        p3d = tde4.getPGroupPosition3D(pg,camera,frame)
                        p3d = convertZup(p3d,yup)
                        r3d = tde4.getPGroupRotation3D(pg,camera,frame)
                        rot = convertToAngles(r3d)
                        if frame>1:
                            rot = [ angleMod360(rot0[0],rot[0]), angleMod360(rot0[1],rot[1]), angleMod360(rot0[2],rot[2]) ]
                        rot0 = rot
                        f.write("setKeyframe -at translateX -t %d -v %.15f $pointGroupName; "%(frame+frame0,p3d[0]))
                        f.write("setKeyframe -at translateY -t %d -v %.15f $pointGroupName; "%(frame+frame0,p3d[1]))
                        f.write("setKeyframe -at translateZ -t %d -v %.15f $pointGroupName; "%(frame+frame0,p3d[2]))
                        f.write("setKeyframe -at rotateX -t %d -v %.15f $pointGroupName; "%(frame+frame0,rot[0]))
                        f.write("setKeyframe -at rotateY -t %d -v %.15f $pointGroupName; "%(frame+frame0,rot[1]))
                        f.write("setKeyframe -at rotateZ -t %d -v %.15f $pointGroupName;\n"%(frame+frame0,rot[2]))

                        frame += 1

                    if export_model == 1:
                        for i in tde4.get3DModelList(pg):
                            model_path = tde4.get3DModelFilepath(pg, i)
                            model_name = tde4.get3DModelName(pg, i)
                            model_name = os.path.splitext(model_name)[0]
                            model_name = validName(model_name)
                            if os.path.isfile(model_path):
                                f.write("file -reference -type \"OBJ\" -loadReferenceDepth \"all\" -mergeNamespacesOnClash true -options \"mo=1\" \"%s\";\n"%(model_path))
                                #f.write("file -import -type \"OBJ\" -ra true -mergeNamespacesOnClash false -rpr \"%s\" -options \"mo=1\" \"%s\";\n"%(model_name, model_path))

            #
            # global (scene node) transformation...

            p3d = tde4.getScenePosition3D()
            p3d = convertZup(p3d,yup)
            r3d = tde4.getSceneRotation3D()
            rot = convertToAngles(r3d)
            s = tde4.getSceneScale3D()
            f.write("xform -zeroTransformPivots -rotateOrder zxy -translation %.15f %.15f %.15f -scale %.15f %.15f %.15f -rotation %.15f %.15f %.15f $sceneGroupName;\n\n"%(p3d[0],p3d[1],p3d[2],s,s,s,rot[0],rot[1],rot[2]))

            f.write("\n")
            f.close()

            path2 = tde4.getWidgetValue(req,"file_browser")
            if not path2.endswith('.mel'): path = path+'.mel'
            f2 = open("/tmp/tde4_exported_mel.txt", "w")
            f2.write(path2)
            f2.close()

            write_log(path2)

            tde4.postQuestionRequester("Export Maya...","Project successfully exported. \n OVERSCAN VALUE : " + str(overscan), "Ok")
        else:
            tde4.postQuestionRequester("Export Maya...","Error, couldn't open file.","Ok")
# except Exception as ex:
#     print ex