import operator

from pxr import Tf
from pxr.Usdviewq.plugin import PluginContainer


def setMask(usdviewApi):
    maskMode = usdviewApi.dataModel.viewSettings.cameraMaskMode
    if maskMode == 'none':
        maskMode = 'partial'
    else:
        maskMode = 'none'
    usdviewApi.dataModel.viewSettings.cameraMaskMode = maskMode

    reticlesInside = usdviewApi.dataModel.viewSettings.showReticles_Inside
    usdviewApi.dataModel.viewSettings.showReticles_Inside = operator.not_(reticlesInside)


class CameraMaskPluginContainer(PluginContainer):

    def registerPlugins(self, plugRegistry, usdviewApi):
        self._setMask = plugRegistry.registerCommandPlugin(
            "CameraMaskPluginContainer.setMask", "Camera Mask", setMask
        )

    def configureView(self, plugRegistry, plugUIBuilder):
        dxMenu = plugUIBuilder.findOrCreateMenu('Dexter')
        dxMenu.addItem(self._setMask)

Tf.Type.Define(CameraMaskPluginContainer)
