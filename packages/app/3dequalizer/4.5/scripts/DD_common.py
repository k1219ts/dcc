import os
import re
import glob
import itertools
import tde4
import TDE4_common
import json
#from tactic_client_lib import TacticServerStub
import pprint
import requests
import dxConfig
from operator import itemgetter, attrgetter

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
showConfigPath = '/dexter/Cache_DATA/MMV/asset/setup/show_config.json'


def run_tactic_api(params):

    params['api_key'] = API_KEY
    infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')), params=params).json()

    return infos

def get_show_name(show):

    project = {}
    project['api_key'] = API_KEY

    if show == "test_shot":
        show = "testshot"

    if show.count('show') > 0:
        project['code'] = show
        infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')), params=project).json()

        return infos[0]['name']

    else:
        project['name'] = show
        infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')), params=project).json()

        return infos[0]['code']


def get_my_task(show, user):

    tasks = []
    params = {}
    params['api_key'] = API_KEY
    params['login'] = user
    # params['login'] = getpass.getuser()

    #print params['login']

    info = requests.get("http://10.0.0.51/dexter/search/task.php", params=params).json()

    #pprint.pprint(info)

    #info2 = sorted(info, key=lambda info: (info['priority'], info['extra_code']), reverse=True)
    #info2 = sorted(info, key=lambda info: (info['priority'], info['project_code']), reverse=True)
    info2 = sorted(sorted(info, key=lambda info: (info['priority']), reverse=True), key=lambda info: (info['milestone_code']))

    tmp = []
    for i in info2:
        #print i['project_code'], i['extra_code'], i['status'], i['priority']
        if i['status'] != 'Approved' and i['status'] != 'Waiting' and i['status'] != 'Omit' and i['status'] != 'Hold'\
                and i['search_type'].count('shot?'):
            a = {}
            a['project_code'] = get_show_name(i['project_code'])
            a['extra_code'] = i['extra_code']
            a['extra_name'] = i['extra_name']
            a['status'] = i['status']
            a['priority'] = i['priority']
            if i['milestone_code']:
                milestone = str(i['milestone_code']).split('_')
                a['milestone_code'] = milestone[0] + "_" + milestone[1]
            else:
                a['milestone_code'] = "None"
            if not a['extra_code'] in tmp:
                tasks.append(a)
                tmp.append(a['extra_code'])

    return tasks

def get_mmv_task(show, shot):

    tasks = ""
    shot_query = {}
    shot_query['api_key'] = API_KEY
    shot_query['project_code'] = get_show_name(show)
    shot_query['code'] = shot

    #print project_code
    #print shot

    infos = requests.get("http://%s/dexter/search/shot.php" %(dxConfig.getConf('TACTIC_IP')), params = shot_query).json()

    try:
        for i in infos[0]['tasks']:
            #print i
            if i['process'] == 'matchmove':
                tasks = tasks + i['context'] + ": " + i['assigned'] + "\n"
                #print i['context'] + ": " + i['assigned']
    except:
        tasks = "there is no Matchmove Task"

    if tasks == "":
        tasks = "there is no Matchmove Task"

    return tasks

def get_shot_info(show, shot):

    shotInfo = ""

    #print shot

    for i in shot:

        params = {}
        params['api_key'] = API_KEY
        params['project_code'] = get_show_name(show)
        params['code'] = i

        try:
            infos = requests.get("http://%s/dexter/search/shot.php" % (dxConfig.getConf('TACTIC_IP')), params=params).json()

            shotInfo = shotInfo + i + ": " + str(infos[0]['frame_in']) + "F ~ " + str(infos[0]['frame_out']) + "F\n"
        except:
            return "Wrong Shot name. check plz."

    return shotInfo

def get_show_list():
    show_list = []
    if os.path.isfile(showConfigPath):
        #print list_path
        f = open(showConfigPath, "r")
        try:
            j = json.load(f)
            show_list = j["show_list"]
            #print show_list
        except:
            raise Exception("Parsing JSON failed! It goes something wrong.")
        finally:
            f.close()
            return show_list

def get_team_list():
    team_list = []
    if os.path.isfile(showConfigPath):
        #print list_path
        f = open(showConfigPath, "r")
        try:
            j = json.load(f)
            team_list = j["team_list"]
            #print show_list
        except:
            raise Exception("Parsing JSON failed! It goes something wrong.")
        finally:
            f.close()
            return team_list

def get_show_config(show, key):
    show_config = []
    if os.path.isfile(showConfigPath):
        #print json_path
        f = open(showConfigPath, "r")
        try:
            j = json.load(f)
            show_config = j[show][key]
            #print show_config
        except:
            raise Exception("Parsing JSON failed! It goes something wrong.")
        finally:
            f.close()
            return show_config

def get_asdDir_list():
    result = []
    all_list = []
    dir = os.path.join("/", "show")
    try:
        all_list = os.listdir(dir)
    except:
        result.append("No Dirs")
        return result

    if all_list:
        for i in all_list:
            if os.path.isdir(os.path.join(dir, i)) and i[0] != ".":
                if i.count("asd") and i != "asd" and i != "asd01" and i != "asd02" and i != "asd03" and i != "asd04" :
                    if not i.count("_pub"):
                        result.append(i)

    result.sort()
    return result

def get_dir_list(dir):
    result = []
    all_list = []
    try:
        all_list = os.listdir(dir)
    except:
        result.append("No Dirs")
        return result

    if all_list:
        for i in all_list:
            if os.path.isdir(os.path.join(dir, i)) and i[0] != ".":
                result.append(i)

    result.sort()
    return result

def reorder_list(list, inx):
    try:
        id = list.index(inx)
        value = list[id]
        list.pop(id)
        list.insert(0, value)
    except:
        print "reorder_list(): There is no %s in list.\n"%inx

    return list

def make_dir(dir):
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
            #print "make_dir done."
            return True
        except:
            import traceback
            var = traceback.format_exc()
            print var
            return False
    else:
        #print "dir exists."
        return True

def get_file_list(path, filename="", ext="*.*"):
    all_list = []
    result = []

    try:
        #os.listdir(path)
        all_list = glob.glob(os.path.join(path, ext))

        if all_list:
            for i in all_list:
                result.append(os.path.basename(i))
            result = [x for x in result if filename in x]
            if not result:
                result = ["No File1"]
        else:
            result = ["No File2"]
    except:
        result = ["No Dir"]
        return result
    
    result.sort()
    return result

def get_final_ver(path="", filename="", ext=""):
    file_list = get_file_list(path, filename, ext)
    file_list.sort()

    if file_list[0].startswith("No"):
        return 0
    else:
        final_file = os.path.splitext(file_list[-1])    # result: ('ELX_0100_main_matchmove_v11', '.mb')
        final_file = final_file[0]    #'ELX_0100_main_matchmove_v11'
        final_ver = final_file.split('_')[-1]    # result: v11
        final_ver = int(final_ver[1:])    # result: 11
        return final_ver

def find_list_item(requester, widget):
    label = ''
    for i in range(tde4.getListWidgetNoItems(requester, widget)):
        if tde4.getListWidgetItemSelectionFlag(requester, widget, i):
            label = tde4.getListWidgetItemLabel(requester, widget, i)
            return label

def find_list_index(requester, widget):
    label = ''
    for i in range(tde4.getListWidgetNoItems(requester, widget)):
        if tde4.getListWidgetItemSelectionFlag(requester, widget, i):
            index = tde4.getListWidgetItemLabel(requester, widget, i)
            return index

def findListItems(requester, widget):
    label = []
    for i in range(tde4.getListWidgetNoItems(requester, widget)):
        if tde4.getListWidgetItemSelectionFlag(requester, widget, i):
            label.append(tde4.getListWidgetItemLabel(requester, widget, i))
    return label

def extractNumber(name):
    num = re.findall(r"\d+", name)
    if len(num) != 0:
        return num[-1]
    else:
        return 0

def getFileList(dirPath):
    allList = glob.glob(os.path.join(dirPath, "*"))
    files = []
    for i in allList:
        if not os.path.isdir(i):
            files.append(os.path.basename(i))
    return sorted(files)

def getSeqFile(group):
    if len(group) == 1:
        return group[0][1]

    fileRoot, fileExt = os.path.splitext(group[0][1])    # result: "SHS_0420_main_v01.0101", ".jpg"

    startFramePadding = extractNumber(fileRoot)    # result: "0101"
    frameIndex = fileRoot.rfind(startFramePadding)    # result: 18
    fileName = fileRoot[:frameIndex]    # result: "SHS_0420_main_v01."

    startFrame = extractNumber(group[0][1])    # result: 0101
    endFrame = extractNumber(group[-1][1])    # result: 0181

    length = len(str(int(endFrame)))    # result: 3
    #padding = "#" * len(startFramePadding)    # result: "####"

    seqFileName = "%s%s%s"%(fileName, startFrame, fileExt)    # result: "SHS_0420_main_v01.0101.jpg"
    #return "%s [%s-%s]" % (seqFileName, startFrame[-length:], endFrame[-length:])
    return seqFileName, startFrame[-length:], endFrame[-length:]

def getSeqFileList(dirPath):
    DATA = getFileList(dirPath)
    seqList = [getSeqFile(tuple(group)) \
        for key, group in itertools.groupby(enumerate(DATA), lambda(index, name): index - int(extractNumber(name)))]
    return seqList
    
def clearProject(mode):
    for c in tde4.getCameraList():
        tde4.deleteCamera(c)
    for p in tde4.getPGroupList():
        tde4.deletePGroup(p)
    for l in tde4.getLensList():
        tde4.deleteLens(l)
    
    if mode:
        newCam=tde4.createCamera("SEQUENCE")
        tde4.createPGroup("CAMERA")
        newLens=tde4.createLens()
        tde4.setCameraLens(newCam, newLens)

    tde4.clearRenderCache()

def checkListItemSelection(req, widget):
    num = tde4.getListWidgetNoItems(req, widget)
    for i in range(num):
        if tde4.getListWidgetItemSelectionFlag(req, widget, i):
            return True
    return False

def importBcompress():
    cam_list = tde4.getCameraList()
    for i in cam_list:
        tde4.importBufferCompressionFile(i)

def exportNukeDewarpNode2(cam, offset):
    lens     = tde4.getCameraLens(cam)
    model     = tde4.getLensLDModel(lens)
    num_frames     = tde4.getCameraNoFrames(cam)
    w_fb_cm = tde4.getLensFBackWidth(lens)
    h_fb_cm = tde4.getLensFBackHeight(lens)
    lco_x_cm = tde4.getLensLensCenterX(lens)
    lco_y_cm = tde4.getLensLensCenterY(lens)
    pxa = tde4.getLensPixelAspect(lens)
    # xa,xb,ya,yb in unit coordinates, in this order.
    fov = tde4.getCameraFOV(cam)

    f = ""
    
    # f += "# Created by 3DEqualizer4 using Export Nuke Distortion Nodes export script\n"
    f += "LD%s {\n"%TDE4_common.nukify_name(model)
    f += " inputs 0\n"
    f += " direction distort\n"

    # write focal length curve if dynamic
    if tde4.getCameraZoomingFlag(cam):
        f += " tde4_focal_length_cm {{curve "
        for frame in range(1,num_frames + 1):
            f += "x%i"%(frame+offset)
            f += " %.7f "%tde4.getCameraFocalLength(cam,frame)
        f += "}}\n"
    # write static focal length else
    else:
        f += " tde4_focal_length_cm %.7f \n"%tde4.getCameraFocalLength(cam,1)
    # write focus distance curve if dynamic
    try:
        if tde4.getCameraFocusMode(cam) == "FOCUS_DYNAMIC":
            f += " tde4_custom_focus_distance_cm {{curve "
            for frame in range(1,num_frames + 1):
                f += "x%i"%(frame+offset)
                f += " %.7f "%tde4.getCameraFocus(cam,frame)
            f += "}}\n"
    except:
        # For 3DE4 Release 1:
        pass
    # write static focus distance else
    else:
        print 'static focus distance'
        try:
            f += " tde4_custom_focus_distance_cm %.7f \n"%tde4.getCameraFocus(cam,1)
        except:
            # For 3DE4 Release 1:
            f += " tde4_custom_focus_distance_cm 100.0 \n"

    # write camera
    f += " tde4_filmback_width_cm %.7f \n"%w_fb_cm
    f += " tde4_filmback_height_cm %.7f \n"%h_fb_cm
    f += " tde4_lens_center_offset_x_cm %.7f \n"%lco_x_cm
    f += " tde4_lens_center_offset_y_cm %.7f \n"%lco_y_cm
    f += " tde4_pixel_aspect %.7f \n"%pxa
    f += " field_of_view_xa_unit %.7f \n"%fov[0]
    f += " field_of_view_ya_unit %.7f \n"%fov[2]
    f += " field_of_view_xb_unit %.7f \n"%fov[1]
    f += " field_of_view_yb_unit %.7f \n"%fov[3]


# write distortion parameters
#
# dynamic distortion
    try:
        dyndistmode    = tde4.getLensDynamicDistortionMode(lens)
    except:
        # For 3DE4 Release 1:
        if tde4.getLensDynamicDistortionFlag(lens) == 1:
            dyndistmode = "DISTORTION_DYNAMIC_FOCAL_LENGTH"
        else:
            dyndistmode = "DISTORTION_STATIC"
    old_api = True
    try:
    # If this fails, the new python API for getLensLDAdjustableParameter will be used.
    # There was a bug until version 1.3, which lead to old_api false, always.
        for para in (getLDmodelParameterList(model)):
            tde4.getLensLDAdjustableParameter(lens, para, 1)
    except:
        old_api = False

    if old_api:
        if dyndistmode=="DISTORTION_DYNAMIC_FOCAL_LENGTH":
            # dynamic focal length (zoom)
            for para in (TDE4_common.getLDmodelParameterList(model)):
                f += " %s {{curve "%TDE4_common.nukify_name(para)
                for frame in range(1,num_frames + 1):
                    focal = tde4.getCameraFocalLength(cam,frame)
                    f += "x%i"%(frame+offset)
                    f += " %.7f "%tde4.getLensLDAdjustableParameter(lens, para, focal)
                f += "}}\n"

        if dyndistmode=="DISTORTION_DYNAMIC_FOCUS_DISTANCE":
            # dynamic focus distance
            for para in (TDE4_common.getLDmodelParameterList(model)):
                f += " %s {{curve "%TDE4_common.nukify_name(para)
                for frame in range(1,num_frames + 1):
                    # Older Releases do not have Focus-methods.
                    try:
                        focus = tde4.getCameraFocus(cam,frame)
                    except:
                        focus = 100.0
                    f += "x%i" % (frame+offset)
                    f += " %.7f "%tde4.getLensLDAdjustableParameter(lens, para, focus)
                f += "}}\n"
    
        # static distortion
        if dyndistmode=="DISTORTION_STATIC":
            for para in (TDE4_common.getLDmodelParameterList(model)):
                f += " %s %.7f \n"%(TDE4_common.nukify_name(para), tde4.getLensLDAdjustableParameter(lens, para, 1))
    else:
    # new API    
        if dyndistmode=="DISTORTION_STATIC":
            #print 'static lens distortion'
            for para in (TDE4_common.getLDmodelParameterList(model)):
                f += " %s %.7f \n"%(TDE4_common.nukify_name(para), tde4.getLensLDAdjustableParameter(lens, para, 1, 1))
        else:
            #print 'dynamic lens distortion,'
            # dynamic
            for para in (TDE4_common.getLDmodelParameterList(model)):
                f += " %s {{curve "%TDE4_common.nukify_name(para)    
                for frame in range(1,num_frames + 1):
                    focal = tde4.getCameraFocalLength(cam,frame)
                    focus = tde4.getCameraFocus(cam,frame)
                    f += "x%i"%(frame+offset)
                    f += " %.7f "%tde4.getLensLDAdjustableParameter(lens, para, focal, focus)    
                f += "}}\n"

    f += " name %s\n"%TDE4_common.decode_entities(tde4.getCameraName(cam))
    f += "}\n"

    return f
