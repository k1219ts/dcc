"""
Maya global options

TODO:

LAST RELEASE:
- 2017.08.26 : RenderSetupInfo re-build
- 2017.08.30 : remove layer apply, unapply
               camera override : must collection name include "camera"
               If camera override, and then defaultRenderLayer want to render,
                must select defaultRenderLayer
- 2017.10.09 : add RelativePath
"""


from config import *

import maya.cmds as cmds
import maya.mel as mel
import maya.app.renderSetup.model.renderSetup as renderSetup


def FrameRange():
    st = cmds.playbackOptions(q=True, min=True)
    et = cmds.playbackOptions(q=True, max=True)
    return '%d-%d' % (st, et)


def MayaProjectPath(*argv):
    if argv:
        mayafile = argv[0]
    else:
        mayafile = cmds.file(q=True, sn=True)

    src = mayafile.split('/')
    chkpoint = -1
    if 'renderScenes' in src:
        chkpoint = src.index('renderScenes')
    elif 'scenes' in src:
        chkpoint = src.index('scenes')

    if chkpoint > -1:
        proj = os.path.join('/', *src[:chkpoint])
    else:
        proj = cmds.workspace(q=True, rd=True)[:-1]
    return proj


def AbsolutePath(source):
    if not source:
        return ''

    if os.path.isabs(source):
        return source
    else:
        proj = MayaProjectPath()
        filepath = os.path.abspath(os.path.join(proj, source))
        filepath = filepath.replace(os.path.sep,'/')
        return filepath

def RelativePath(source):
    if not source:
        return ''

    if not os.path.isabs(source):
        return source

    proj = MayaProjectPath()
    return source.replace(proj, '.')


def LastVersion():
    proj = MayaProjectPath()
    outputs = list()
    imagepath = os.path.join(proj, 'images')
    if os.path.exists(imagepath):
        for i in os.listdir(imagepath):
            p = re.compile(r'v\d\d\d').findall(i)
            if p:
                outputs.append(p[0])

    outputs.sort()
    if outputs:
        last_version = int(outputs[-1][1:])
    else:
        last_version = 0
    new_version = 'v%03d' % (last_version + 1)
    # print new_version
    return new_version

def VersionList():
    proj = MayaProjectPath()
    outputs = list()
    imagepath = os.path.join(proj, 'images')
    if os.path.exists(imagepath):
        r = re.compile(r'v\d\d\d')
        return filter(r.match, os.listdir(imagepath))

    outputs.sort()
    return outputs

def CurrentVersion():
    # for recovery render image folder
    proj = MayaProjectPath()
    outputs = list()
    imagepath = os.path.join(proj, 'images')
    if os.path.exists(imagepath):
        for i in os.listdir(imagepath):
            p = re.compile(r'v\d\d\d').findall(i)
            if p:
                outputs.append(p[0])

    outputs.sort()
    if outputs:
        last_version = int(outputs[-1][1:])
    else:
        last_version = 0
    current_version = 'v%03d' % (last_version)
    # print new_version
    return current_version


def RenderCameras():
    result = list()
    for i in cmds.ls(type='camera'):
        adjustments = cmds.listConnections('%s.renderable' % i, type='renderLayer', plugs=True)
        if adjustments:
            result.append(i)
        if cmds.getAttr('%s.renderable' % i):
            result.append(i)
    return list(set(result))


def StereoRenderStatus():
    state = False
    cameras = RenderCameras()
    if len(cameras) > 1:
        for i in cameras:
            if i.find('left') > -1 or i.find('right') > -1:
                state = True
    return state


def IterateFrame( frameRange, limit ):
    result = []
    for i in frameRange.split(','):
        if len(i.split('-')) > 1:
            source = i.split('-')
            start_frame = int(source[0])
            end_frame   = int(source[-1])

            hostbyframe = (end_frame - start_frame + 1) / limit
            chk_point   = hostbyframe * limit + start_frame - 1

            if hostbyframe > 1:
                for x in range(limit):
                    sf = start_frame + (x * hostbyframe)
                    ef = sf + hostbyframe - 1
                    if x == limit - 1:
                        result.append( (sf, end_frame) )
                    else:
                        result.append( (sf, ef) )
            else:
                for x in range(start_frame, end_frame+1):
                    result.append( (x, x) )
        else:
            result.append( (int(i), int(i)) )
    return result


class RenderSetupInfo:
    """
    Get RenderSetup Info
    """
    def __init__(self):
        self._info = dict() # final data
        self._doLayer = True
        self.m_default = None # defaultRenderLayer
        self.m_layers = list()

        self.rs = renderSetup.instance()
        self.doIt()

    def doIt(self):
        self.getRenderLayers()
        self.getInfo()

        if not self.m_layers:
            self._doLayer = False


    def getRenderLayers(self):
        # default
        layer = self.rs.getDefaultRenderLayer()
        if layer.isRenderable():
            self.m_default = layer
        # layers
        for layer in self.rs.getRenderLayers():
            self.m_layers.append(layer)


    def getInfo(self):
        if self.m_default:
            cameras = self.getRenderCameras()
            self.updateInfo('defaultRenderLayer', 'camera', cameras)

        for layer in self.m_layers:
            if layer.isRenderable():
                # print '# layer : ', layer
                # camera
                cameras = self.getRenderLayerCameras(layer)
                # print '# cameras : ', cameras
                if cameras:
                    self.updateInfo(layer.name(), 'camera', cameras)
                # render global override
                overrides = self.getRenderSettingOverride(layer)
                if overrides:
                    for i in overrides:
                        self.updateInfo(layer.name(), i, overrides[i])

    def updateInfo(self, Layer, Key, Value):
        if not self._info.has_key(Layer):
            self._info[Layer] = dict()
        if Value:
            self._info[Layer][Key] = Value


    def getRenderCameras(self):
        result = list()
        for i in cmds.ls(type='camera'):
            if cmds.getAttr('%s.renderable' % i):
                result.append(i)
        return result

    def getRenderLayerCameras(self, layer):
        ifever = False
        for collection in layer.getCollections():
            # print '# collection : ', collection.name()
            if collection.name().find('camera') > -1:
                ifever = True
                for override in collection.getOverrides():
                    if override.attributeName() == 'renderable' and override.getAttrValue():
                        cameras = list()
                        for i in collection.getSelector().getAbsoluteNames():
                            cameras.append(i.split('|')[-1])
                        return cameras
        if not ifever:
            return self.getRenderCameras()


    def getRenderSettingOverride(self, layer):
        if not layer.hasRenderSettingsCollectionInstance():
            return
        result = dict()
        collection = layer.renderSettingsCollectionInstance() # RenderSettingsCollection
        for child in collection.getChildren():
            ln = child.attributeName()
            if ln == 'rman__torattr___denoise':
                denoise = cmds.getAttr('renderManRISGlobals.rman__torattr___denoise')
                filter  = cmds.getAttr('renderManRISGlobals.rman__torattr___denoiseFilter')
                result['denoise'] = (denoise, filter)
            if ln == 'rman__riopt__Hider_minsamples':
                if not result.has_key('sampling'):
                    result['sampling'] = dict()
                result['sampling']['minsamples'] = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_minsamples')
            if ln == 'rman__riopt__Hider_maxsamples':
                if not result.has_key('sampling'):
                    result['sampling'] = dict()
                result['sampling']['maxsamples'] = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_maxsamples')
            if ln == 'rman__riopt__Hider_incremental':
                if not result.has_key('sampling'):
                    result['sampling'] = dict()
                result['sampling']['incremental'] = cmds.getAttr('renderManRISGlobals.rman__riopt__Hider_incremental')
        return result


    def getDenoiseInfo(self, ui_denoise, ui_denoiseFilter):
        if not self._doLayer:
            return
        denoiseLayers = list()
        for layer in self._info:
            if self._info[layer].has_key('denoise'):
                d, f = self._info[layer]['denoise']
                if d:
                    denoiseLayers.append('%s:%s:%s' % (d, self.getRibname(layer), f))
            else:
                if ui_denoise:
                    denoiseLayers.append('%s:%s:%s' % (ui_denoise, self.getRibname(layer), ui_denoiseFilter))
        return denoiseLayers


    def getRibname(self, layername):
        if layername == 'defaultRenderLayer':
            return 'masterLayer'
        else:
            return 'rs_%s' % layername


def MayaFilePathSetup(options):
    """
    m_mayafile, m_mayaproj, m_rmsprod, m_outdir : setup
    """
    if not options.has_key('m_mayafile'):
        options['m_mayafile'] = cmds.file(q=True, sn=True)
    if not options.has_key('m_mayaproj'):
        options['m_mayaproj'] = MayaProjectPath()
    if not options.has_key('m_rmsprod'):
        src = options['m_mayaproj'].split('/')
        if 'show' in src:
            options['m_rmsprod'] = string.join(src[:src.index('show')+2], '/')
        else:
            options['m_rmsprod'] = mel.eval('rman getvar RMSPROD')
    # output path
    if not options.has_key('m_version'):
        options['m_version'] = LastVersion()
    if not options.has_key('m_outdir'):
        options['m_outdir'] = os.path.join(options['m_mayaproj'], 'images', options['m_version'])

    # file data
    src = os.path.splitext(os.path.basename(options['m_mayafile']))
    options['m_mayaext'] = src[-1]
    filename = src[0]
    filename = re.sub('_v\d+', '', filename)
    filename = re.sub('_w\d+', '', filename)
    options['m_mayabasename'] = filename
    return options


def MayaFrameRange(options):
    """
    m_range, m_by
    """
    if not options.has_key('m_range'):
        options['m_range'] = FrameRange()
    if not options.has_key('m_by'):
        options['m_by'] = 1
    return options


__DefaultOptions = {
    'm_engine': '10.0.0.35', 'm_port': 80,
    'm_renderer': 'renderManRIS', 'm_priority': 100,
    'm_envkey': str(mel.eval('rman getPref DefaultEnvKey')),
    'm_maxactive': 0,
    'm_user': getpass.getuser(),
    'm_ribgenLimit': 1, 'm_ribgenOnly': 0,
    'm_shutterAngle': 180, 'm_motionBlur': 1, 'm_cameraBlur': 1, 'm_tracedBlur': 1,
    'm_minsamples': 0, 'm_maxsamples': 16, 'm_PixelVariance': 0.1,
    'm_checkPoint': 20, 'm_incremental': 1,
    'm_denoise': 0, 'm_denoiseFilter': 'default.filter.json',
    'm_denoiseaov': 0, 'm_denoiseStrength': 0.2
}

def OptionsSetup(options):
    result = dict()
    for d in (__DefaultOptions, options):
        for key, value in d.items():
            result[key] = value
    # file setup
    result = MayaFilePathSetup(result)
    # frame range
    result = MayaFrameRange(result)
    return result


__all__ = [
    'cmds', 'mel',
    'FrameRange', 'MayaProjectPath', 'AbsolutePath', 'RelativePath', 'LastVersion',
    'CurrentVersion', 'VersionList',
    'RenderCameras', 'StereoRenderStatus', 'IterateFrame',
    'RenderSetupInfo', '__DefaultOptions', 'OptionsSetup'
]

