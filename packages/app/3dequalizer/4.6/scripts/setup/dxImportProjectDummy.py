import os
import tde4
import DD_common
reload(DD_common)


class dxImportDummy:
    def __init__(self, requester):
        self.req = requester

        self.SHOW_ROOT = '/show'
        self.show = os.environ['show']

        self.selDummys = []
        self.windowTitle = None

        self.globalPath = os.path.join(self.SHOW_ROOT, self.show, 'works', 'MMV', 'asset', 'dummy')
        self.assetPath = os.path.join(self.SHOW_ROOT, self.show, '_3d', 'asset', '{asset}')
        self.getAssetList()

    def _assetListCallback(self, requester, widget, action):
        tde4.removeAllListWidgetItems(requester, 'assetType')
        asset = DD_common.find_list_item(requester, 'assetList')

        if asset == 'global':
            dummyList = DD_common.getDummyList(self.globalPath)

            tde4.removeAllListWidgetItems(requester, 'fileList')
            for i in dummyList:
                if i[:1] != '_' and i.count('obj') > 0:
                    count = 0
                    tde4.insertListWidgetItem(requester, 'fileList', i, count)
                    count += 1

        else:
            assetDir = DD_common.get_dir_list(self.assetPath.format(asset=asset))

            # print assetDir

            for i in assetDir:
                count = 0
                tde4.insertListWidgetItem(requester, 'assetType', i, count)
                count += 1

            tde4.removeAllListWidgetItems(requester, 'fileList')
            tde4.insertListWidgetItem(requester, 'fileList', 'Select Asset Type.', 0)

        if tde4.getListWidgetNoItems(requester, 'assetType') == 0:
            tde4.insertListWidgetItem(requester, 'assetType', 'No DIRs.', 0)

    def _assetTypeCallback(self, requester, widget, action):
        chk = DD_common.find_list_item(requester, 'assetType')
        if chk.startswith('No ') or chk.startswith('Select '):
            return
        asset = DD_common.find_list_item(requester, 'assetList')
        assetType = DD_common.find_list_item(requester, 'assetType')

        tmp = os.path.join(self.assetPath.format(asset=asset), assetType)
        ver = DD_common.GetLastVersion(tmp)
        path = os.path.join(tmp, ver)
        file_list = DD_common.getFileList(path, '', '*.obj')

        tde4.removeAllListWidgetItems(requester, 'fileList')
        if file_list[0].startswith('No '):
            file_list = ['No Obj file']
        for i in file_list:
            count = 0
            tde4.insertListWidgetItem(requester, 'fileList', i, count)

    def _fileListCallback(self, requester, widget, action):
        try:
            path = self.getPath()

            ovr = 0
            for i in range(tde4.getListWidgetNoItems(requester, 'selList')):
                label = tde4.getListWidgetItemLabel(requester, 'selList', i)
                if label == path[1]:
                    ovr = 1

            if ovr != 1:
                self.selDummys.append(path)
                tde4.insertListWidgetItem(requester, 'selList', path[1], 0)
            else:
                tde4.postQuestionRequester(self.windowTitle, 'Already selected Asset', 'Ok')
        except:
            tde4.postQuestionRequester(self.windowTitle, 'Error, select Asset file first.', 'Ok')

    def _btnDeleteCallback(self, requester, widget, action):
        deleteAsset = DD_common.find_list_item(requester, 'selList')

        idx = 0
        if deleteAsset:
            #print 'deleteAsset', deleteAsset
            for i in self.selDummys:
                if deleteAsset in i[1]:
                    #print idx
                    tde4.removeListWidgetItem(requester, 'selList', idx)
                    self.selDummys.pop(idx)

                idx += 1

            #print 'selDummys', self.selDummys
        else:
            tde4.postQuestionRequester(self.windowTitle, 'Error, select Asset file first.', 'Ok')

    def getPath(self):
        asset = DD_common.find_list_item(self.req, 'assetList')
        assetType = DD_common.find_list_item(self.req, 'assetType')
        slDummy = DD_common.find_list_item(self.req, 'fileList')

        if asset == 'global':
            path = os.path.join(self.globalPath, slDummy)
        else:
            tmp = os.path.join(self.assetPath.format(asset=asset), assetType)
            ver = DD_common.GetLastVersion(tmp)
            path = os.path.join(tmp, ver, slDummy)

        output = [path, slDummy, assetType]
        return output

    def getAssetList(self):
        assetList = []
        assetPath = os.path.join(self.SHOW_ROOT, self.show, '_3d', 'asset')

        assetList = os.listdir(assetPath)

        for asset in assetList:
            if not '.' in asset:
                for assetType in os.listdir(os.path.join(assetPath, asset)):
                    if assetType in ['lidar', 'pmodel']:
                        tde4.insertListWidgetItem(self.req, 'assetList', asset, 0)
                        break

    def doIt(self):
        if self.selDummys:
            # try:
            tde4.postProgressRequesterAndContinue(self.windowTitle, 'import Asset ...', len(self.selDummys), 'Stop')
            for idx, i in enumerate(self.selDummys):
                cont = tde4.updateProgressRequester(idx, 'Import Asset ... ' + str(i[1]))
                if not cont: break

                pg = tde4.getCurrentPGroup()
                m = tde4.create3DModel(pg, 0)
                tde4.importOBJ3DModel(pg, m, i[0])
                tde4.set3DModelName(pg, m, i[1])
                tde4.set3DModelReferenceFlag(pg, m, 1)
                tde4.set3DModelSurveyFlag(pg, m, 1)
                if i[2] == 'lidar':
                    tde4.set3DModelPerformanceRenderingFlag(pg, m, 1)
            # except:
            #     print 'import Error'

            tde4.unpostProgressRequester()
        else:
            tde4.postQuestionRequester(self.windowTitle, 'Error, select Asset file first.', 'Ok')
