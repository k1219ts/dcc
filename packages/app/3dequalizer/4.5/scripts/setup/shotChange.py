#
# 3DE4.script.name:  shotChange Tool ...
#
# 3DE4.script.version:  v1.0.0
#
# 3DE4.script.gui:  Main Window::Dexter
#
# 3DE4.script.comment:  Shot change Tool.
#
# Written by kwantae.kim(kkt0525@gmail.com)

#
# import modules and get environments
import os
import shutil
import DD_common
reload(DD_common)
global show, seq, shot, platetype

# user-defined functions...
def _pubTool_callback(requester, widget, action):

    show = os.environ["show"]

    if widget == "toShot":

        try:
            shot = tde4.getWidgetValue(requester, "toShot")

            seq = shot.split('_')[0]

            #print seq, shot

            shotInfo = DD_common.get_shot_info(os.environ["show"], [os.environ["shot"], shot])

            tde4.clearTextAreaWidget(requester, "shotInfo")
            tde4.appendTextAreaWidgetString(requester, "shotInfo", shotInfo)

            shotPath = os.path.join("/", "show", show, "shot", seq, shot, "plates")

            # print shotPath

            platesList = DD_common.get_dir_list(shotPath)
            verList = DD_common.get_dir_list(shotPath)

            tde4.removeAllListWidgetItems(requester, "platetype")
            for i in platesList:
                count = 0
                tde4.insertListWidgetItem(req, "platetype", i, count)
                count += 1
        except:
            tde4.clearTextAreaWidget(requester, "shotInfo")
            tde4.appendTextAreaWidgetString(requester, "shotInfo", "input shotName.")

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

def copyMayaScene(toPath):

    shot_root = os.path.join(SHOW_ROOT, os.environ["show"], "shot", os.environ["seq"], os.environ["shot"])
    mmv_root = os.path.join(shot_root, "matchmove")

    from_pubPath = os.path.join(mmv_root, "pub", "scenes")

    if os.path.isdir(from_pubPath):
        i = os.listdir(from_pubPath)

        #print len(i)

        try:
            if len(i) > 1:
                i.sort(reverse=True)

            lastVer = i[0]

            #print from_pubPath, lastVer

            from_pubPath = os.path.join(from_pubPath, lastVer)
            from_fileList = DD_common.get_file_list(from_pubPath, "", "*.mb")
            fromPath = os.path.join(from_pubPath, from_fileList[0])

            print "from: " + fromPath
            print "to: " + toPath

            shutil.copy(fromPath, toPath)
        except:
            print 'MayaScene Copy Error!'


def writeShotinfo():

    os.environ["show"] = show
    os.environ["seq"] = seq
    os.environ["shot"] = shot
    os.environ["platetype"] = platetype

# user-defined variables...
window_title = "shotChange Tool v1.0.0..."
SHOW_ROOT = "/show"

#
# main...

if os.environ.has_key('show'):

    # open requester...
    req = tde4.createCustomRequester()

    tde4.addTextFieldWidget(req, "show", "show", os.environ["show"])
    tde4.setWidgetSensitiveFlag(req, "show", 0)

    tde4.addTextFieldWidget(req, "fromShot", "from shot", os.environ["shot"])
    tde4.setWidgetSensitiveFlag(req, "fromShot", 0)

    tde4.addSeparatorWidget(req, "sep01")

    tde4.addTextFieldWidget(req, "toShot", "to shot", "")
    tde4.setWidgetCallbackFunction(req, "toShot", "_pubTool_callback")

    tde4.addTextAreaWidget(req, "shotInfo", "shot Info", 100, 0)

    tde4.addListWidget(req, "platetype", "Plate Type", 0, 70)
    tde4.insertListWidgetItem(req, "platetype", "Input Shot.", 0)

    tde4.addToggleWidget(req, 'add_plate', 'add Plate', 1)
    #tde4.addToggleWidget(req, 'make_dir', 'make Dir', 1)
    tde4.addToggleWidget(req, 'copy_mayaScene', 'Copy mayaScene', 1)

    # tde4.addSeparatorWidget(req, 'sep02')
    # tde4.addTextAreaWidget(req, "result", "Result", 100, 0)

    ret = tde4.postCustomRequester(req, window_title, 600, 420, "Save", "Cancel")

    if ret == 1:
        show = os.environ["show"]
        shot = tde4.getWidgetValue(req, "toShot")
        seq = shot.split('_')[0]
        platetype = DD_common.find_list_item(req, "platetype")

        if platetype!=None:
            if not platetype.startswith("Input ") or platetype.startswith("No "):    gogo = 1
        else:    gogo = 0

        if gogo == 0:
            tde4.postQuestionRequester(window_title, "Select plateType.", "Ok")
        else:
            shot_root = os.path.join(SHOW_ROOT, show, "shot", seq, shot)
            shot_mmv_root = os.path.join(shot_root, "matchmove")

            new_shot_path = os.path.join(shot_mmv_root, "dev", "3de", shot + "_%s" % platetype + "_matchmove_v01_01.3de")

            #print new_shot_path

            if os.path.isfile(new_shot_path):
                path = os.path.split(new_shot_path)[0]  # /show/prat/shot/SHK/SHK_1780/matchmove/dev/3de
                file = os.path.split(new_shot_path)[1]  # SHK_1780_main_matchmove_v01_01.3de
                fileName = file.split(".")[0]  # SHK_1780_main_matchmove_v01_01
                splitFileName = fileName.split("_")  # ['SHK', '1780', 'main', matchmove', 'v01', '01']
                newReVer = int(splitFileName[-1]) + 1  # 01 + 1
                splitFileName[-1] = "%.2d" % newReVer  # ['SHK', '1780', 'main', matchmove', 'v01', '02']
                newFileName = "_".join(splitFileName)  # SHK_1780_main_matchmove_v01_02

                new_shot_path = os.path.join(path, newFileName+".3de")
                #print new_shot_path

            msg = "'%s'\nDo you want to Save the project?" % new_shot_path
            ans = tde4.postQuestionRequester(window_title, msg, "Yes", "No")

            if ans == 1:
                if not os.path.isdir(shot_mmv_root):
                    make_mmv_dirs(shot_mmv_root)

                if tde4.getWidgetValue(req, "copy_mayaScene") == 1:
                    copyMayaScene(os.path.join(shot_mmv_root, "dev", "scenes", shot + "_%s" % platetype + "_matchmove_v01_w01.mb"))

                if tde4.getWidgetValue(req, "add_plate") == 1:
                    shotPath = os.path.join("/", "show", show, "shot", seq, shot, "plates", platetype)
                    slVersion = DD_common.get_dir_list(shotPath)
                    slVersion.sort(reverse=True)
                    fileList = DD_common.getSeqFileList(os.path.join(shotPath, slVersion[0]))

                    #print shotPath, "\n", platetype, "\n", slVersion[0]

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

                        #print fileName, "\n", fileName2

                        cam = tde4.createCamera("SEQUENCE")
                        if gamma:
                            tde4.setCamera8BitColorGamma(cam, float(gamma))
                        tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName))
                        tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName2))
                        tde4.setCameraName(cam, camName)
                        tde4.setCameraSequenceAttr(cam, int(start[1:]), int(end[:-1]), 1)

                tde4.saveProject(new_shot_path)
                writeShotinfo()

            else:
                tde4.postQuestionRequester(window_title, "Cancelled.", "Ok")
else:
    tde4.postQuestionRequester(window_title, "There is no \"ENVKEY\"\nPlease open a project using \"Open Project\" script first.", "Ok")
