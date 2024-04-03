import operator

from pxr import Tf, Usd, UsdSkel
from pxr.Usdviewq.plugin import PluginContainer


def changePurpose(usdviewApi):
    proxy = usdviewApi.dataModel.viewSettings.displayProxy
    usdviewApi.dataModel.viewSettings.displayProxy = operator.not_(proxy)
    render = usdviewApi.dataModel.viewSettings.displayRender
    usdviewApi.dataModel.viewSettings.displayRender = operator.not_(render)


class PurposeChangePluginContainer(PluginContainer):

    def registerPlugins(self, plugRegistry, usdviewApi):
        self._changePurpose = plugRegistry.registerCommandPlugin(
            "PurposeChangePluginContainer.changePurpose", "Change Purpose", changePurpose
        )

    def configureView(self, plugRegistry, plugUIBuilder):
        dxMenu = plugUIBuilder.findOrCreateMenu('Dexter')
        dxMenu.addItem(self._changePurpose)

Tf.Type.Define(PurposeChangePluginContainer)
