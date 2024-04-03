import subprocess
import os
import sys


def modelExport(usdviewApi,orgDir,newShow,orgAssetName,assetName,Element,overwrite,orgMDir):
    var = usdviewApi.selectedPrims[0].GetVariantSets().GetAllVariantSelections()
    purposeList = []
    for i in usdviewApi.selectedPrims[0].GetChildren():
        if i.GetName() == "render":
            purposeList.append(i)
        elif i.GetName() == "proxy":
            purposeList.append(i)
        else:
            pass

    if purposeList:
        Purpose = True
    else:
        Purpose = False

    if 'lodVariant' in var.keys():
        Lod = True
    else:
        Lod = False

    if 'modelVersion' in var.keys():
        # print 'orgMDir:  ',orgMDir
        orgModelDir = "{MODELDIR}/{VER}".format(MODELDIR=orgMDir,VER = var["modelVersion"])
    else:
        print 'orgModelDir is None'
        pass

    if 'zennVersion' in var.keys():
        orgZennVer = var['zennVersion']
        print 'Get AssetInfo:   Get zenn Version:   ',orgZennVer,'\n\n'

        # cmdPath = '/WORK_DATA/develop/apps/USD-toolkit/view-plugins/assetExportPlugin/mayaCmd_zenn.py'
        cmdPath = '/backstage/apps/USD-toolkit/view-plugins/assetExportPlugin/mayaCmd_zenn.py'

        zennCommand = ['/backstage/bin/DCC', 'mayapy',
                       cmdPath,
                       '--orgDir', orgDir,
                       '--orgModelDir', orgModelDir,
                       '--orgZennVer', orgZennVer,
                       '--newShow', newShow,
                       '--orgAssetName', orgAssetName,
                       '--assetName', assetName,
                       '--Element', str(Element),
                       '--Purpose', str(Purpose),
                       '--Lod', str(Lod),
                       '--overwrite', str(overwrite)
                       ]

        zennCmd = ' '.join(zennCommand)
        print 'Progress     :   ',zennCmd,'\n\n'

        if newShow == '/assetlib/3D':
            print 'assetlib Export'
            suCmd = 'echo dexter2019 | su render -c "%s"' % zennCmd
            subprocess.Popen(suCmd, shell=True).wait()
        else:
            subprocess.Popen(zennCmd, shell=True).wait()

    else:
        # cmdPath ='/WORK_DATA/develop/apps/USD-toolkit/view-plugins/assetExportPlugin/mayaCmd_base.py'
        cmdPath ='/backstage/apps/USD-toolkit/view-plugins/assetExportPlugin/mayaCmd_base.py'

        command = ['/backstage/bin/DCC', 'mayapy',
                   cmdPath,
                   '--orgDir', orgDir,
                   '--orgModelDir', orgModelDir,
                   '--newShow', newShow,
                   '--orgAssetName', orgAssetName,
                   '--assetName', assetName,
                   '--Element', str(Element),
                   '--Purpose', str(Purpose),
                   '--Lod', str(Lod),
                   '--overwrite', str(overwrite)
                   ]

        cmd = ' '.join(command)
        print 'Progress     :   ',cmd,'\n\n'

        if newShow == '/assetlib/3D':
            print 'assetlib Export'
            suCmd = 'echo dexter2019 | su render -c "%s"' % cmd
            subprocess.Popen(suCmd, shell=True).wait()
        else:
            subprocess.Popen(cmd, shell=True).wait()

    print 'Progress     :   Model Export successed'








