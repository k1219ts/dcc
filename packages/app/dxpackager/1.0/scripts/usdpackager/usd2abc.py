# coding:utf-8
import sys
import os
import shutil
from time import time

import json
import re

import maya.standalone
import maya.cmds as cmds

maya.standalone.initialize()

import dbutils

class ExportAbc:

    OPTIONS = ";shadingMode=none;readAnimData=1;assemblyRep=Import"
    RE_MB = re.compile(r"\.m[ab]$", re.I)
    RE_SHOW = re.compile("/(show|mach)/")
    RE_LOW_MB = re.compile(r"_(low|mid)(?P<ext>\.m[ab])$", re.I)

    ABC_ARGS = ("-framerange {} {} -renderableonly -selection -uvwrite -writeuvsets "
                "-writevisibility -autosubd -dataformat ogawa -file {}")

    def __init__(self, order_path):
        print("[usd2abc] Processing...")

        self.time0 = time()

        self.startFrame = None
        self.endFrame = None
        self.frameRate = None

        self.packageFolder = None
        self.sourcePath = None
        self.framePath = None

        self.logPath = None
        self.log = []

        self.setup(order_path)
        self.do_job()

    def setup(self, order_path):
        if not os.path.isfile(order_path):
            sys.exit(os.EX_OSERR)

        for plugin in ["pxrUsd", "pxrUsdTranslators", "AbcImport", "AbcExport"]:
            if not cmds.pluginInfo(plugin, q=True, loaded=True):
                if cmds.loadPlugin(plugin):
                    print("{} is loaded.".format(plugin))
                else:
                    cmds.error("{} cannot be loaded.".format(plugin))
                    sys.exit()

        with open(order_path, 'r') as argv_f:
            order_data = json.load(argv_f)

        self.startFrame = order_data["start_frame"]
        self.endFrame = order_data["end_frame"]
        self.frameRate = order_data["fps"]

        self.logPath = log_path = order_data["log_path"]
        self.packageFolder = package_folder = order_data["package_folder"]
        self.sourcePath = order_data["shot_path"]
        self.framePath = order_data["frame_path"]
        self.abcPath = order_data["abc_path"]

        if not os.path.exists(package_folder):
            os.makedirs(package_folder)

        print("[usd2abc] order path : {}".format(order_path))
        print("[usd2abc] package folder : {}".format(package_folder))
        print("[usd2abc] report path : {}".format(log_path))

    def do_job(self):
        print("[usd2abc] Preparing Export...")

        for path in [self.framePath, self.packageFolder]:
            if self.RE_SHOW.match(path):
                self.report("- invalid path : {}".format(path))
                return self.exit()

        self.setup_time()
        self.export_frame_range()

        self.report(self.sourcePath)
        cmds.file(self.sourcePath, i=True, type="pxrUsdImport", options=self.OPTIONS)
        self.export_abc()
        dbutils.updatePackage(self.sourcePath, self.abcPath, {'pkgType': 'abc'})
        shotDir = os.path.dirname(self.sourcePath)
        aniDir = shotDir+'/ani'
        if os.path.isdir(aniDir):
            try: lsnslyr = os.listdir(aniDir)
            except: lsnslyr = []
            rlsnslyr = sorted(lsnslyr, reverse=True)
            for nslyr in rlsnslyr:
                nslyrDir = aniDir+'/'+nslyr
                try: lsver = os.listdir(nslyrDir)
                except: lsver = []
                rlsver = sorted(lsver, reverse=True)
                for ver in rlsver:
                    verDir = nslyrDir+'/'+ver
                    verUsd = verDir+'/'+nslyr+'_ani.usd'
                    if bool(re.match(r'^v[0-9]{3}', ver)):
                        dbutils.updatePackage(self.sourcePath, self.abcPath, {'pkgType': 'abc', 'file': verUsd })
                        break
        self.exit()

    TIME_UNITS = [
        ("game", 15),
        ("film", 24),
        ("pal", 25),
        ("ntsc", 30),
        ("show", 48),
        ("palf", 50),
        ("ntscf", 60),
        ("23.976fps", 23.976),
        ("29.97fps", 29.97),
        ("59.94fps", 59.94)
    ]

    def setup_time(self):
        frame_rate = self.frameRate
        for time_unit, fps in self.TIME_UNITS:
            if abs(frame_rate - fps) < 0.001:
                cmds.currentUnit(time=time_unit)
                break
        cmds.playbackOptions(animationStartTime=self.startFrame, min=self.startFrame,
                             animationEndTime=self.endFrame, max=self.endFrame)

    def export_frame_range(self):
        print("[usd2abc] saving frame range {}...".format(self.framePath))
        cmds.file(rename=self.framePath)
        try:
            cmds.file(s=True, f=True)
        except:
            self.report("- failed time-ranged scene")
        else:
            self.report("+ time-ranged scene : {}".format(self.framePath))

        # to save frame range to a json
        json_path = self.RE_MB.sub(".json", self.framePath)
        try:
            with open(json_path, 'w') as f:
                json.dump({"startFrame": self.startFrame, "endFrame": self.endFrame}, f)
        except:
            self.report("- failed time range data")
        else:
            self.report("+ time range data : {}".format(json_path))

    def export_abc(self):
        cmds.select(cmds.ls("World", dag=True))
        try:
            cmds.AbcExport(j=self.ABC_ARGS.format(self.startFrame, self.endFrame, self.abcPath))
        except:
            self.report("- failed abc export.")
        else:
            self.report("+ -> {}".format(self.abcPath))

    def report(self, message):
        print("[usd2abc] {}".format(message))
        self.log.append(message)

    def exit(self):
        m, s = divmod(time() - self.time0, 60)
        collapsed_time = "{} min. {:.0f} sec.".format(int(m), s)
        self.report("End of Process : {}".format(collapsed_time))

        if self.log and self.logPath:
            with open(self.logPath, 'w') as f:
                f.write('\n'.join(self.log))
        sys.exit()


if __name__ == '__main__':
    ExportAbc(sys.argv[1])
