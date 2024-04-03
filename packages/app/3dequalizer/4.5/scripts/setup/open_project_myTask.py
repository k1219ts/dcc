#
# 3DE4.script.name:  1-2. Open Project myTasks ...
#
# 3DE4.script.version:  v1.1.1
#
# 3DE4.script.gui:  Main Window::DD_Setup
#
# 3DE4.script.comment:  Open Project myTasks.
#
# Written by kwantae.kim(kkt0525@gmail.com)

#
# import modules and get environments
import os
import getpass
import DD_common
reload(DD_common)

global user, show, seq, shot, platetype

# user-defined functions...
def _open_project_callback(requester, widget, action):

    if widget == 'userlist':

        tde4.removeAllListWidgetItems(requester, "shotlist")

        mode = tde4.getWidgetValue(requester, "userlist") - 1
        user = user_list[mode]

        shot_list = DD_common.get_my_task("", user)

        idx = 0
        for i in shot_list:
            # print i['extra_code']
            count = 0
            if i['priority'] == None:
                i['priority'] = 0
            tmp = "[" + i['project_code'].upper() + "] " + i['extra_code'] + "  .." + str(i['status']) + "  " + str(
                i['milestone_code']) + "  " + str(i['priority'])
            tde4.insertListWidgetItem(req, "shotlist", tmp, count)
            # if i['priority'] == 5:
            #    tde4.setListWidgetItemColor(req, "shotlist", idx, 1, 1, 0)
            if i['status'] == "Ready":
                tde4.setListWidgetItemColor(req, "shotlist", idx, 1, 1, 0.6)
            elif i['status'] == "In-Progress":
                tde4.setListWidgetItemColor(req, "shotlist", idx, 0.75, 1, 0.15)
            elif i['status'] == "OK":
                tde4.setListWidgetItemColor(req, "shotlist", idx, 0, 1, 1)
            elif i['status'] == "Retake":
                tde4.setListWidgetItemColor(req, "shotlist", idx, 1, 0, 0)
            elif i['status'] == "Review":
                tde4.setListWidgetItemColor(req, "shotlist", idx, 0, 0.8, 0)
            count += 1
            idx += 1

        if len(shot_list) == 0:
            tde4.insertListWidgetItem(req, "shotlist", "No Tasks.", 0)
        return

    shot = DD_common.find_list_item(req, "shotlist")

    info = shot.split(' ')

    show = info[0].replace("[", "").replace("]", "").lower()
    shot = info[1]
    seq = info[1].split("_")

    if seq[0].count("pos") > 0:
        seq[0] = "POS"

    if show == 'testshot':
        show = 'test_shot'

    if widget == "shotlist":

        chk = DD_common.find_list_item(requester, "shotlist")
        #print chk
        if chk.startswith("No ") or chk.startswith("Select "):
            return

        shotPath = os.path.join("/", "show", show, "shot", seq[0], shot, "plates")
        platesList = DD_common.get_dir_list(shotPath)

        #print shotPath, platesList

        tde4.removeAllListWidgetItems(requester, "platetype")

        for i in platesList:
            count = 0
            tde4.insertListWidgetItem(req, "platetype", i, count)
            count += 1
        tde4.removeAllListWidgetItems(requester, "filelist")
        tde4.insertListWidgetItem(requester, "filelist", "Select Plate Type.", 0)

    if widget == "platetype":
        chk = DD_common.find_list_item(requester, "platetype")
        if chk.startswith("No ") or chk.startswith("Select "):
            return

        platetype = DD_common.find_list_item(req, "platetype")
        file_list = DD_common.get_file_list(
            os.path.join(SHOW_ROOT, show, "shot", seq[0], shot, "matchmove", "dev", "3de"),
            platetype,
            "*.3de")
        file_list.sort(reverse=True)
        tde4.removeAllListWidgetItems(requester, "filelist")
        if file_list[0].startswith("No "):
            file_list = ["Create New Project."]
        for i in file_list:
            count = 0
            tde4.insertListWidgetItem(requester, "filelist", i, count)
            count += 1


def make_mmv_dirs(dir):
    """
    <default folder structure for matchmove department>
    /matchmove/dev/
              |   /3de
              |   /imageplane
              |   /nuke
              |   /preview
              |   /scenes
              /pub/
                  /imageplane
                  /nuke
                  /preview
                  /scenes
    """

    phase = ["dev", "pub"]
    subset = ["3de", "nuke", "scenes", "preview", "imageplane"]

    for i in phase:
        if i == "pub":
            subset.pop(0)
        for k in subset:
            DD_common.make_dir(os.path.join(dir, i, k))


def writeShotinfo():
    os.environ["show"] = show
    os.environ["seq"] = seq[0]
    os.environ["shot"] = shot
    os.environ["platetype"] = platetype


# user-defined variables...
window_title = "Open Project myTasks v1.1.1..."
SHOW_ROOT = "/show"
shot_list = DD_common.get_my_task("", getpass.getuser())

#
# main...

# open requester...
req = tde4.createCustomRequester()

user_list = []
user_list.append(getpass.getuser())

team_list = DD_common.get_team_list()

for i in team_list:
    if i != getpass.getuser():
        user_list.append(i)

tde4.addOptionMenuWidget(req, "userlist", "User", *user_list)
tde4.setWidgetCallbackFunction(req, "userlist", "_open_project_callback")
#tde4.setListWidgetItemSelectionFlag(req, "userlist", user_list.index(user), 1)

tde4.addListWidget(req, "shotlist", "Shot", 0, 400)
tde4.setWidgetCallbackFunction(req, "shotlist", "_open_project_callback")
idx = 0
for i in shot_list:
    # print i['extra_code']
    count = 0
    if i['priority'] == None:
        i['priority'] = 0
    tmp = "[" + i['project_code'].upper() + "] " + i['extra_name'] + "  .." + str(i['status']) + "  " + str(
        i['milestone_code']) + "  " + str(i['priority'])
    tde4.insertListWidgetItem(req, "shotlist", tmp, count)
    # if i['priority'] == 5:
    #    tde4.setListWidgetItemColor(req, "shotlist", idx, 1, 1, 0)
    if i['status'] == "Ready":
        tde4.setListWidgetItemColor(req, "shotlist", idx, 1, 1, 0.6)
    elif i['status'] == "In-Progress":
        tde4.setListWidgetItemColor(req, "shotlist", idx, 0.75, 1, 0.15)
    elif i['status'] == "OK":
        tde4.setListWidgetItemColor(req, "shotlist", idx, 0, 1, 1)
    elif i['status'] == "Retake":
        tde4.setListWidgetItemColor(req, "shotlist", idx, 1, 0, 0)
    elif i['status'] == "Review":
        tde4.setListWidgetItemColor(req, "shotlist", idx, 0, 0.8, 0)
    count += 1
    idx += 1

if len(shot_list) == 0:
    tde4.insertListWidgetItem(req, "shotlist", "No Tasks.", 0)

tde4.addListWidget(req, "platetype", "Plate Type", 0, 60)
tde4.insertListWidgetItem(req, "platetype", "Select Shot.", 0)
tde4.setWidgetCallbackFunction(req, "platetype", "_open_project_callback")

tde4.addListWidget(req, "filelist", "File", 0, 100)
tde4.insertListWidgetItem(req, "filelist", "Select Plate Type.", 0)

tde4.addToggleWidget(req, "import_cache", "Import Image Cache", 0)

ret = tde4.postCustomRequester(req, window_title, 600, 690, "Open", "Cancel")

if ret == 1:
    shot = DD_common.find_list_item(req, "shotlist")

    info = shot.split(' ')
    show = info[0].replace("[", "").replace("]", "").lower()
    shot = info[1]
    seq = info[1].split("_")

    if seq[0].count("pos") > 0:
        seq[0] = "POS"

    if show == 'testshot':
        show = 'test_shot'

    # print show, seq[0], shot

    platetype = DD_common.find_list_item(req, "platetype")
    filename = DD_common.find_list_item(req, "filelist")

    if show != None:
        if not show.startswith("Select ") or show.startswith("No "):    gogo = 1
    else:
        gogo = 0
    if shot != None:
        if not shot.startswith("Select ") or shot.startswith("No "):    gogo = 1
    else:
        gogo = 0
    if platetype != None:
        if not platetype.startswith("Select ") or platetype.startswith("No "):    gogo = 1
    else:
        gogo = 0
    if filename != None:
        if not filename.startswith("Select ") or filename.startswith("No "):    gogo = 1
    else:
        gogo = 0

    if gogo == 0:
        tde4.postQuestionRequester(window_title, "Select Sequence or Shot or File.", "Ok")
    else:
        shot_root = os.path.join(SHOW_ROOT, show, "shot", seq[0], shot)
        shot_mmv_root = os.path.join(shot_root, "matchmove")

        projPath = tde4.getProjectPath()
        # print projPath
        if not projPath == None:
            ans = tde4.postQuestionRequester(window_title, "Do you want to Save the project?", "Yes", "No")
            if ans == 1:
                tde4.saveProject(projPath)
            #else:
                #tde4.postQuestionRequester(window_title, "save Cancelled.", "Ok")

        if filename == "Create New Project.":

            if not os.path.isdir(shot_mmv_root):
                ans = tde4.postQuestionRequester(window_title,
                                                 "There is no matchmove folder in shot folder. Create matchmove folder?",
                                                 "Yes", "No")
                if ans == 1:
                    make_mmv_dirs(shot_mmv_root)
                    DD_common.clearProject(1)

                    shotPath = os.path.join("/", "show", show, "shot", seq[0], shot, "plates", platetype)
                    slVersion = DD_common.get_dir_list(shotPath)
                    slVersion.sort(reverse=True)
                    fileList = DD_common.getSeqFileList(os.path.join(shotPath, slVersion[0]))
                    # print shotPath, "\n", platetype, "\n", slVersion[0]

                    cam = ""
                    gamma = DD_common.get_show_config(show, 'gamma')

                    for i in fileList:
                        tmp = "%s :[%s-%s]" % (i[0], i[1], i[2])
                        # print tmp
                        fileName, frameRange = tmp.split(" :")  # result: "SHS_0420_main_v02.0101.jpg", "[101-103]"
                        start, end = frameRange.split("-")  # result: "[101", "103]"
                        num = DD_common.extractNumber(fileName)  # result: 0101
                        pad = "#" * len(num)  # result: "####"
                        fileName2 = fileName.replace(num, pad)  # result: "SHS_0420_main_v02.####.jpg"

                        frameIndex = fileName.rfind(num)  # result: 18
                        camName = fileName[:frameIndex - 1]  # result: "SHS_0420_main_v01"

                        # print fileName, "\n", fileName2

                        cam = tde4.createCamera("SEQUENCE")
                        if gamma:
                            tde4.setCamera8BitColorGamma(cam, float(gamma))
                        tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName))
                        tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName2))
                        tde4.setCameraName(cam, camName)
                        tde4.setCameraSequenceAttr(cam, int(start[1:]), int(end[:-1]), 1)

                    # /show/prs/shot/_source/MMV_Cache/BEL/BEL_0100/plates/main1/v01
                    cacheDIR = os.path.join(SHOW_ROOT, show, "shot/_source/MMV_Cache", seq[0], shot)

                    if os.path.isdir(cacheDIR):
                        cache_Path = os.path.join(cacheDIR, "plates", platetype)
                        slVersion = DD_common.get_dir_list(cache_Path)
                        slVersion.sort(reverse=True)
                        fileList = DD_common.getSeqFileList(os.path.join(cache_Path, slVersion[0]))

                        # print cache_Path

                        for i in fileList:
                            tmp = "%s :[%s-%s]" % (i[0], i[1], i[2])
                            # print tmp
                            fileName, frameRange = tmp.split(" :")  # result: "SHS_0420_main_v02.0101.jpg", "[101-103]"
                            start, end = frameRange.split("-")  # result: "[101", "103]"
                            num = DD_common.extractNumber(fileName)  # result: 0101
                            pad = "#" * len(num)  # result: "####"
                            fileName2 = fileName.replace(num, pad)  # result: "SHS_0420_main_v02.####.jpg"

                            frameIndex = fileName.rfind(num)  # result: 18
                            camName = fileName[:frameIndex - 1]  # result: "SHS_0420_main_v01"

                            # print fileName, "\n", fileName2

                            tde4.setCameraProxyFootage(cam, 1)

                            if gamma:
                                tde4.setCamera8BitColorGamma(cam, float(gamma))
                            tde4.setCameraPath(cam, os.path.join(cache_Path, slVersion[0], fileName))
                            tde4.setCameraPath(cam, os.path.join(cache_Path, slVersion[0], fileName2))
                            tde4.setCameraName(cam, camName)
                            tde4.setCameraSequenceAttr(cam, int(start[1:]), int(end[:-1]), 1)

                        #tde4.postQuestionRequester(window_title, "2K plate imported. check Proxy Footage.", "Ok")

                    tde4.saveProject(os.path.join(shot_mmv_root, "dev", "3de", shot + "_%s" % platetype + "_matchmove_v01_01.3de"))
                    writeShotinfo()
                else:
                    tde4.postQuestionRequester(window_title, "Cancelled.", "Ok")
            else:
                DD_common.clearProject(1)
                tde4.saveProject(
                    os.path.join(shot_mmv_root, "dev", "3de", shot + "_%s" % platetype + "_matchmove_v01_01.3de"))
                writeShotinfo()

        else:
            prj = os.path.join(shot_mmv_root, "dev", "3de", filename)
            DD_common.clearProject(0)
            tde4.loadProject(prj)
            writeShotinfo()
            if tde4.getWidgetValue(req, "import_cache") == 1:
                DD_common.importBcompress()
