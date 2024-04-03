'''
Dexter USD Pipeline Tools
'''
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

import maya.cmds as cmds

from PackageUtils import CleanUpSubLayers
from PathUtils import GetVersion, GetProjectPath
from Model import ModelExport, ModelClipExport
from EnvSet import GetSetNodes, SetAssetExport, SetAssetImport, SetShotExport, TaneSourceTextureOverride
from Rig import GetRigNodes, RigAssetExport, RigShotExport, RigClipExport
from Sim import GetSimNodes, SimExport
from Camera import GetCameraNodes, CameraAssetExport, CameraShotExport
from Zenn import ZennAssetExport, ZennShotExport
from Texture import TextureExport, AssetTextureOverride
from Asset import GetTempAssetPath, TempAssetExport
from ClipUtils import ClipEdit
from Light import LightInstanceExport

__all__ = [
    "CleanUpSubLayers", "GetVersion", "GetProjectPath",
    "ModelExport", "ModelClipExport",
    "GetSetNodes", "SetAssetExport", "SetAssetImport", "SetShotExport", "TaneSourceTextureOverride",
    "GetRigNodes", "RigAssetExport", "RigShotExport", "RigClipExport",
    "GetSimNodes", "SimExport",
    "GetCameraNodes", "CameraAssetExport", "CameraShotExport",
    "ZennAssetExport", "ZennShotExport",
    "TextureExport", "AssetTextureOverride",
    "LightInstanceExport",
    "TempAssetExport",
    "ClipEdit",
]


if cmds.pluginInfo('MiarmyProForMaya2017', q=True, l=True) or cmds.pluginInfo('MiarmyProForMaya2018', q=True, l=True):
    from Crowd import AgentExport, CrowdShotExport
    modules = ["AgentExport", "CrowdShotExport"]
    __all__ += modules
