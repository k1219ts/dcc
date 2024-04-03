#
# 3DE4.script.name:  1-3. Open Project for Asdal...
#
# 3DE4.script.version:  v1.7.0
#
# 3DE4.script.gui:  Main Window::DD_Setup
#
# 3DE4.script.comment:  Open Project for Asdal.
#

#
# import modules and get environments
import os
import DD_common
reload(DD_common)
global show, seq, shot, platetype

# user-defined functions...
def _open_project_callback(requester, widget, action):

    mode = tde4.getWidgetValue(requester, "showlist") - 1
    show = show_list[mode]

    if widget == "showlist":
        if show.startswith("No ") or show.startswith("Select "):
            return
        seq_list = DD_common.get_dir_list(os.path.join(SHOW_ROOT, show, "shot"))

        tde4.removeAllListWidgetItems(requester, "seqlist")
        for i in seq_list:
            count = 0
            tde4.insertListWidgetItem(requester, "seqlist", i, count)
            count += 1
        tde4.removeAllListWidgetItems(requester, "shotlist")
        tde4.insertListWidgetItem(requester, "shotlist", "Select Sequence.", 0)
        tde4.removeAllListWidgetItems(requester, "platetype")
        tde4.insertListWidgetItem(requester, "platetype", "Select Shot.", 0)
        tde4.removeAllListWidgetItems(requester, "filelist")
        tde4.insertListWidgetItem(requester, "filelist", "Select Plate Type.", 0)
    if widget == "seqlist":
        chk = DD_common.find_list_item(requester, "seqlist")
        if chk.startswith("No ") or chk.startswith("Select "):
            return
        tde4.removeAllListWidgetItems(requester, "shotlist")
        seq = DD_common.find_list_item(requester, "seqlist")
        shot_list = DD_common.get_dir_list(os.path.join(SHOW_ROOT, show, "shot", seq))
        for i in shot_list:
            count = 0
            tde4.insertListWidgetItem(requester, "shotlist", i, count)
            count += 1
        tde4.removeAllListWidgetItems(requester, "platetype")
        tde4.insertListWidgetItem(requester, "platetype", "Select Shot.", 0)
        tde4.removeAllListWidgetItems(requester, "filelist")
        tde4.insertListWidgetItem(requester, "filelist", "Select Plate Type.", 0)
    if widget == "shotlist":
        chk = DD_common.find_list_item(requester, "shotlist")
        if chk.startswith("No ") or chk.startswith("Select "):
            return
        seq = DD_common.find_list_item(req, "seqlist")
        shot = DD_common.find_list_item(req, "shotlist")
        shotPath = os.path.join("/", "show", show, "shot", seq, shot, "plates")
        #shotPath = os.path.join("/", "show", os.environ["show"], "shot", os.environ["seq"], os.environ["shot"], "plates")
        platesList = DD_common.get_dir_list(shotPath)
        #verList = DD_common.get_dir_list(shotPath)

        tde4.removeAllListWidgetItems(requester, "platetype")
        tde4.clearTextAreaWidget(requester, "task_user")
        for i in platesList:
            count = 0
            tde4.insertListWidgetItem(req, "platetype", i, count)
            count += 1
        tde4.removeAllListWidgetItems(requester, "filelist")
        tde4.insertListWidgetItem(requester, "filelist", "Select Plate Type.", 0)

        #task user info
        tasks = DD_common.get_mmv_task(show, shot)
        tde4.clearTextAreaWidget(requester, "task_user")
        tde4.appendTextAreaWidgetString(requester, "task_user", tasks)

    if widget == "platetype":
        chk = DD_common.find_list_item(requester, "platetype")
        if chk.startswith("No ") or chk.startswith("Select "):
            return
        seq = DD_common.find_list_item(req, "seqlist")
        shot = DD_common.find_list_item(req, "shotlist")
        platetype = DD_common.find_list_item(req, "platetype")
        file_list = DD_common.get_file_list(os.path.join(SHOW_ROOT, show, "shot", seq, shot, "matchmove", "dev", "3de"),
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
    os.environ["seq"] = seq
    os.environ["shot"] = shot
    os.environ["platetype"] = platetype

# user-defined variables...
window_title = "Open Project for Asdal v1.7.0..."
SHOW_ROOT = "/show"
#show_list = DD_common.get_show_list()
show_list = DD_common.get_asdDir_list()

#print show_list

seq_list = DD_common.get_dir_list(os.path.join(SHOW_ROOT, show_list[0], "shot"))

#
# main...

# open requester...
req = tde4.createCustomRequester()
tde4.addOptionMenuWidget(req, "showlist", "Show", *show_list)
tde4.setWidgetCallbackFunction(req, "showlist", "_open_project_callback")

tde4.addListWidget(req, "seqlist", "Sequence", 0, 100)
for i in seq_list:
    count = 0
    tde4.insertListWidgetItem(req, "seqlist", i, count)
    count += 1
tde4.setWidgetCallbackFunction(req, "seqlist", "_open_project_callback")

tde4.addListWidget(req, "shotlist", "Shot", 0, 200)
tde4.insertListWidgetItem(req, "shotlist", "Select Sequence.", 0)
tde4.setWidgetCallbackFunction(req, "shotlist", "_open_project_callback")

tde4.addListWidget(req, "platetype", "Plate Type", 0, 60)
tde4.insertListWidgetItem(req, "platetype", "Select Shot.", 0)
tde4.setWidgetCallbackFunction(req, "platetype", "_open_project_callback")

tde4.addListWidget(req, "filelist", "File", 0, 100)
tde4.insertListWidgetItem(req, "filelist", "Select Plate Type.", 0)

tde4.addToggleWidget(req, "import_cache", "Import Image Cache", 0)
tde4.addSeparatorWidget(req, "sep01")

tde4.addTextAreaWidget(req, "task_user", "Task User", 100, 0)

ret = tde4.postCustomRequester(req, window_title, 600, 700, "Open", "Cancel")

if ret == 1:
    show = show_list[tde4.getWidgetValue(req, "showlist")-1]
    seq = DD_common.find_list_item(req, "seqlist")
    shot = DD_common.find_list_item(req, "shotlist")
    platetype = DD_common.find_list_item(req, "platetype")
    filename = DD_common.find_list_item(req, "filelist")

    if show!=None:
        if not show.startswith("Select ") or show.startswith("No "):    gogo = 1
    else:    gogo = 0
    if seq!=None:
        if not seq.startswith("Select ") or seq.startswith("No "):    gogo = 1
    else:    gogo = 0
    if shot!=None:
        if not shot.startswith("Select ") or shot.startswith("No "):    gogo = 1
    else:    gogo = 0
    if platetype!=None:
        if not platetype.startswith("Select ") or platetype.startswith("No "):    gogo = 1
    else:    gogo = 0
    if filename!=None:
        if not filename.startswith("Select ") or filename.startswith("No "):    gogo = 1
    else:    gogo = 0

    if gogo == 0:
        tde4.postQuestionRequester(window_title, "Select Sequence or Shot or File.", "Ok")
    else:
        shot_root = os.path.join(SHOW_ROOT, show, "shot", seq, shot)
        shot_mmv_root = os.path.join(shot_root, "matchmove")

        projPath = tde4.getProjectPath()
        #print projPath
        if not projPath == None:
            ans = tde4.postQuestionRequester(window_title, "Do you want to Save the project?", "Yes", "No")
            if ans == 1:
                tde4.saveProject(projPath)
            #else:
                #tde4.postQuestionRequester(window_title, "save Cancelled.", "Ok")

        if filename == "Create New Project.":

            if not os.path.isdir(shot_mmv_root):
                ans = tde4.postQuestionRequester(window_title, "There is no matchmove folder in shot folder. Create matchmove folder?", "Yes", "No")
                if ans == 1:
                    make_mmv_dirs(shot_mmv_root)
                    DD_common.clearProject(1)

                    shotPath = os.path.join("/", "show", show, "shot", seq, shot, "plates", platetype)
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

                    # /show/prs/shot/_source/MMV_Cache/BEL_0100/plates/main1/v01
                    cacheDIR = os.path.join(SHOW_ROOT, show, "shot/_source/MMV_Cache", seq[0], shot)
                    print cacheDIR

                    if os.path.isdir(cacheDIR):
                        cache_Path = os.path.join(cacheDIR, "plates", platetype)
                        slVersion = DD_common.get_dir_list(cache_Path)
                        slVersion.sort(reverse=True)
                        fileList = DD_common.getSeqFileList(os.path.join(cache_Path, slVersion[0]))

                        #print cache_Path

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
                tde4.saveProject(os.path.join(shot_mmv_root, "dev", "3de", shot + "_%s"%platetype + "_matchmove_v01_01.3de"))
                writeShotinfo()

        else:
            prj = os.path.join(shot_mmv_root, "dev", "3de", filename)
            DD_common.clearProject(0)
            tde4.loadProject(prj)
            writeShotinfo()
            if tde4.getWidgetValue(req, "import_cache") == 1:
                DD_common.importBcompress()
