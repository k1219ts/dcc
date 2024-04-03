
from pxr import Tf
from pxr.Usdviewq.plugin import PluginContainer

import dialog
import Main

def main(usdviewApi):
    #getName
    orgAssetName = usdviewApi.selectedPrims[0].GetName()
    getNewInfo =[]
    for i in dialog.dialogWindow(usdviewApi):
        getNewInfo.append(i)

    newShow = getNewInfo[0]
    newAssetName = getNewInfo[1]
    Element = getNewInfo[2]
    overwrite = getNewInfo[3]
    tag=getNewInfo[4]


    if newAssetName == ''and Element== '' and overwrite == ''and tag == '': #cancel button clicked
        print "Export canceled"
    else:
        Main.doit(usdviewApi,orgAssetName,newShow,newAssetName,Element,overwrite,tag)

class assetExportPluginContainer(PluginContainer):

    def registerPlugins(self, plugRegistry, usdviewApi):
        self._export = plugRegistry.registerCommandPlugin(
            "assetExportPluginContainer.export", "Export Asset", main)

    def configureView(self, plugRegistry, plugUIBuilder):
        dxMenu = plugUIBuilder.findOrCreateMenu('Dexter')
        dxMenu.addItem(self._export)

Tf.Type.Define(assetExportPluginContainer)

