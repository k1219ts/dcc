#
#
# 3DE4.script.name:  4. Import Project Asset...
#
# 3DE4.script.version:  v1.3.0
#
# 3DE4.script.gui:  Main Window::DD_Setup
#
# 3DE4.script.comment:  Import Project Asset OBJ.
#
#
# Written by kwantae.kim(kkt0525@gmail.com)

#
# import modules and get environments
import os
import DD_common
reload(DD_common)
global show, seq, shot, platetype

assetType = ["env", "vehicle"]
selDummys = []

# user-defined functions...
def _import_dummy_callback(requester, widget, action):

    if widget == "assetList":
        tde4.removeAllListWidgetItems(requester, "assetType")
        asset = DD_common.find_list_item(requester, "assetList")

        if asset == "global":
            globalPath = os.path.join(SHOW_ROOT, show, "asset", "global", "matchmove", "dummy")
            dummyList = DD_common.getFileList(globalPath)

            # print dummyList

            tde4.removeAllListWidgetItems(requester, "fileList")
            for i in dummyList:
                if i[:1] != "_" and i.count('obj') > 0:
                    count = 0
                    tde4.insertListWidgetItem(req, "fileList", i, count)
                    count += 1

        else:
            selAsset = str(asset).split(" ")

            #print selAsset

            selAsset[0] = selAsset[0].replace("[", "")
            selAsset[0] = selAsset[0].replace("]", "")

            assetDir = DD_common.get_dir_list(os.path.join(SHOW_ROOT, show, "asset", selAsset[0], selAsset[1]))

            #print assetDir

            for i in assetDir:
                count = 0
                tde4.insertListWidgetItem(requester, "assetType", i, count)
                count += 1

            tde4.removeAllListWidgetItems(requester, "fileList")
            tde4.insertListWidgetItem(requester, "fileList", "Select Asset Type.", 0)

        if tde4.getListWidgetNoItems(requester, "assetType") == 0:
            tde4.insertListWidgetItem(requester, "assetType", "No DIRs.", 0)

    if widget == "assetType":
        chk = DD_common.find_list_item(requester, "assetType")
        if chk.startswith("No ") or chk.startswith("Select "):
            return
        asset = DD_common.find_list_item(req, "assetList")
        assetType = DD_common.find_list_item(req, "assetType")

        selAsset = str(asset).split(" ")

        selAsset[0] = selAsset[0].replace("[", "")
        selAsset[0] = selAsset[0].replace("]", "")

        file_list = DD_common.get_file_list(os.path.join(SHOW_ROOT, show, "asset", selAsset[0], selAsset[1], assetType, "pub", "scenes"), "", "*.obj")

        tde4.removeAllListWidgetItems(requester, "fileList")
        if file_list[0].startswith("No "):
            file_list = ["No Obj file"]
        for i in file_list:
            count = 0
            tde4.insertListWidgetItem(requester, "fileList", i, count)

    if widget == "fileList":
        try:
            path = get_path()

            ovr = 0
            for i in range(tde4.getListWidgetNoItems(requester, "selList")):
                label = tde4.getListWidgetItemLabel(requester, "selList", i)
                if label == path[1]:
                    ovr = 1

            if ovr != 1:
                selDummys.append(path)
                tde4.insertListWidgetItem(req, "selList", path[1], 0)
            else:
                tde4.postQuestionRequester(window_title, "Already selected Asset", "Ok")
        except:
            tde4.postQuestionRequester(window_title, "Error, select Asset file first.", "Ok")

    if widget == "btnDelete":
        deleteAsset = DD_common.find_list_item(req, "selList")

        idx = 0
        if deleteAsset:
            #print "deleteAsset", deleteAsset
            for i in selDummys:
                if deleteAsset in i[1]:
                    #print idx
                    tde4.removeListWidgetItem(requester, "selList", idx)
                    selDummys.pop(idx)

                idx += 1

            #print "selDummys", selDummys
        else:
            tde4.postQuestionRequester(window_title, "Error, select Asset file first.", "Ok")


def get_path():
    asset = DD_common.find_list_item(req, "assetList")
    assetType = DD_common.find_list_item(req, "assetType")
    slDummy = DD_common.find_list_item(req, "fileList")

    selAsset = str(asset).split(" ")

    selAsset[0] = selAsset[0].replace("[", "")
    selAsset[0] = selAsset[0].replace("]", "")

    if asset == "global":
        path = os.path.join(SHOW_ROOT, show, "asset", "global", "matchmove", "dummy", slDummy)
    else:
        path = os.path.join(SHOW_ROOT, show, "asset", selAsset[0], selAsset[1], assetType, "pub", "scenes", slDummy)

    output = [path, slDummy, assetType]
    return output

# user-defined variables...
window_title = "Import Project Asset v1.3.0"
SHOW_ROOT = "/show"

#
# main...
if os.environ.has_key('show'):

    show = os.environ["show"]
    if show.count('asd') > 0:
        show = 'asd'

    # open requester...
    req = tde4.createCustomRequester()

    tde4.addTextFieldWidget(req, "show", "Show", show)
    tde4.setWidgetSensitiveFlag(req, "show", 0)

    tde4.addListWidget(req, "assetList", "Asset List", 0, 200)
    tde4.insertListWidgetItem(req, "assetList", "global", 0)

    for t in assetType:
        env_list = DD_common.get_dir_list(os.path.join(SHOW_ROOT, show, "asset", t))

        for i in env_list:
            count = 0
            tde4.insertListWidgetItem(req, "assetList", "[" + t + "] " + i, count)
            count += 1
    tde4.setWidgetCallbackFunction(req, "assetList", "_import_dummy_callback")

    tde4.addListWidget(req, "assetType", "Asset Type", 0, 80)
    tde4.insertListWidgetItem(req, "assetType", "Select Asset.", 0)
    tde4.setWidgetCallbackFunction(req, "assetType", "_import_dummy_callback")

    tde4.addListWidget(req, "fileList", "Asset File", 0, 150)
    tde4.insertListWidgetItem(req, "fileList", "Select Asset Type.", 0)
    tde4.setWidgetCallbackFunction(req, "fileList", "_import_dummy_callback")

    tde4.addButtonWidget(req, "btnDelete", "Delete", 100, 470)
    tde4.setWidgetCallbackFunction(req, "btnDelete", "_import_dummy_callback")

    tde4.addListWidget(req, "selList", "selected Asset File", 0, 100)

    ret = tde4.postCustomRequester(req, window_title, 600, 0, "Import", "Cancel")

    cnt = 0
    if ret == 1:
        if selDummys:
            tde4.postProgressRequesterAndContinue(window_title, 'import Asset ...', len(selDummys), 'Stop')
            for i in selDummys:
                try:
                    cnt += 1
                    cont = tde4.updateProgressRequester(cnt, "Import Asset ... " + str(i[1]))
                    if not cont: break

                    pg = tde4.getCurrentPGroup()
                    m = tde4.create3DModel(pg, 0)
                    tde4.importOBJ3DModel(pg, m, i[0])
                    tde4.set3DModelName(pg, m, i[1])
                    tde4.set3DModelReferenceFlag(pg, m, 1)
                    tde4.set3DModelSurveyFlag(pg, m, 1)
                    if i[2] == "lidar":
                        tde4.set3DModelPerformanceRenderingFlag(pg, m, 1)
                except:
                    print "import Error"

            tde4.unpostProgressRequester()
        else:
            tde4.postQuestionRequester(window_title, "Error, select Asset file first.", "Ok")

else:
    tde4.postQuestionRequester(window_title,
                                   "There is no \"ENVKEY\"\nPlease open a project using \"Open Project\" script first.",
                                   "Ok")