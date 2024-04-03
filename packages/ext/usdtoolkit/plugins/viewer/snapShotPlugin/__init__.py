from pxr import Tf
from pxr.Usdviewq.plugin import PluginContainer
import os
import glob
import re
import datetime
import subprocess

def viewerSetup(usdviewApi):
    usdviewApi._UsdviewApi__appController._mainWindow.setGeometry(0, 0, 1920, 1280)
    topHeight, bottomHeight = usdviewApi._UsdviewApi__appController._ui.topBottomSplitter.sizes()
    primViewWidth, stageViewWidth = usdviewApi._UsdviewApi__appController._ui.primStageSplitter.sizes()
    if bottomHeight > 0 or primViewWidth > 0:  # is toggle mode
        topHeight += bottomHeight
        bottomHeight = 0
        stageViewWidth += primViewWidth
        primViewWidth = 0
        usdviewApi._UsdviewApi__appController._ui.topBottomSplitter.setSizes([topHeight, bottomHeight])
        usdviewApi._UsdviewApi__appController._ui.primStageSplitter.setSizes([primViewWidth, stageViewWidth])

    usdviewApi._UsdviewApi__appController._stageView.setGeometry(0, 0, 1280, 720)
    usdviewApi._UsdviewApi__appController._stageView.updateGeometry()
    usdviewApi._UsdviewApi__appController._stageView.updateGL()


def curFrameSnapShot(usdviewApi):
    viewerSetup(usdviewApi)

    filepath = usdviewApi.stage.GetRootLayer().realPath
    directory = os.path.dirname(filepath)
    basename = os.path.basename(filepath).split('.')[0]

    previewDir = os.path.join(directory, "preview")
    print '# Debug preview dir :', previewDir

    if os.path.exists(previewDir):
        try:
            # versionList = sorted(glob.glob('{0}/*v*'.format(previewDir)))
            # lastestVersion = os.path.basename(versionList[-1])
            res = [f for f in sorted(os.listdir(previewDir)) if re.search(r'v[0-9]{3}', f)]
            lastestVersion = res[-1]
            versionCount = int(lastestVersion.split('_v')[-1].replace('.mov', ''))
            # versionCount = int(lastestVersion[1:])
        except:
            versionCount = 0
    else:
        versionCount = 0

    snapshotPath = os.path.join(directory, "preview", "v%03d" % (versionCount + 1), "%s.jpg" % basename)
    img = usdviewApi.GrabViewportShot()

    if not os.path.exists(os.path.dirname(snapshotPath)):
        os.makedirs(os.path.dirname(snapshotPath))
    img.save(snapshotPath)

    usdviewApi._UsdviewApi__appController._toggleViewerMode()


def frameRangeSnapShot(usdviewApi):
    viewerSetup(usdviewApi)

    filepath = usdviewApi.stage.GetRootLayer().realPath
    directory = os.path.dirname(filepath)
    basename = os.path.basename(filepath).split('.')[0]

    previewDir = os.path.join(directory, "preview")
    print '# Debug preview dir :', previewDir

    if os.path.exists(previewDir):
        try:
            # versionList = sorted(glob.glob('{0}/*v*.mov'.format(previewDir)))
            # lastestVersion = os.path.basename(versionList[-1])
            # versionCount = int(lastestVersion[1:])
            res = [f for f in sorted(os.listdir(previewDir)) if re.search(r'v[0-9]{3}', f)]
            lastestVersion = res[-1]
            versionCount = int(lastestVersion.split('_v')[-1].replace('.mov', ''))
        except:
            versionCount = 0
    else:
        versionCount = 0

    # startFrame = usdviewApi.stage.GetStartTimeCode()
    # endFrame = usdviewApi.stage.GetEndTimeCode()
    startFrame = float(usdviewApi._UsdviewApi__appController._ui.rangeBegin.text())
    endFrame = float(usdviewApi._UsdviewApi__appController._ui.rangeEnd.text())
    FPS = usdviewApi.stage.GetFramesPerSecond()

    format = "%d%m%y-%H%M%S"
    timestamp = datetime.datetime.now().strftime(format)
    snapshotDirPath = os.path.join(directory, "preview", "v%03d" % (versionCount + 1))
    if not os.path.exists(snapshotDirPath):
        os.makedirs(snapshotDirPath)

    for frame in range(int(startFrame), int(endFrame)):
        # frame = usdviewApi._UsdviewApi__appController._ui.frameField.text()
        usdviewApi._UsdviewApi__appController._ui.frameField.setText(str(frame))
        usdviewApi._UsdviewApi__appController._frameStringChanged()
        snapShotPath = os.path.join(snapshotDirPath, "%s.%04d.jpg" % (basename, frame))
        usdviewApi.GrabViewportShot().save(snapShotPath)

    # make mov
    # cmd = ["/opt/ffmpeg/bin/ffmpeg"]
    cmd = ['/backstage/bin/DCC', 'rez-env', 'ffmpeg-4.2.0', '--', 'ffmpeg']
    cmd += ["-r", str(FPS)]
    cmd += ["-start_number", str(startFrame)]
    cmd += ["-i", os.path.join(snapshotDirPath, basename + ".%04d.jpg"), "-an"]
    cmd += ["-r", str(FPS)]
    cmd += ["-vcodec", "libx264"]
    cmd += ["-pix_fmt", "yuv420p", "-preset", "slow", "-profile:v", "baseline"]
    cmd += ["-b", "6000k", "-tune", "zerolatency"]
    cmd += ["-y", "%s/%s_%s.mov" % (os.path.dirname(snapshotDirPath), basename, "v%03d" % (versionCount + 1))]

    os.system(" ".join(cmd))
    usdviewApi._UsdviewApi__appController._toggleViewerMode()
    os.system('rm -rf %s' % snapshotDirPath)

class SnapshotPluginContainer(PluginContainer):

    def registerPlugins(self, plugRegistry, usdviewApi):
        self._curFrameSnapshot = plugRegistry.registerCommandPlugin(
            "SnapshotPluginContainer.curFrameSnapshot", "Current Frame Snapshot", curFrameSnapShot
        )
        self._fullFrameSnapshot = plugRegistry.registerCommandPlugin(
            "SnapshotPluginContainer.fullFrameSnapshot", "Full Frame Snapshot", frameRangeSnapShot
        )

    def configureView(self, plugRegistry, plugUIBuilder):
        dxMenu = plugUIBuilder.findOrCreateMenu("Dexter")
        dxMenu.addItem(self._curFrameSnapshot)
        dxMenu.addItem(self._fullFrameSnapshot)

Tf.Type.Define(SnapshotPluginContainer)
