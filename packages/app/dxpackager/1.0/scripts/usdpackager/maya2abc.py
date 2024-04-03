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


class ExportAbc:
    RE_MB = re.compile(r"\.m[ab]$", re.I)
    RE_SHOW = re.compile("/(show|mach)/")
    RE_LOW_MB = re.compile(r"_(low|mid)(?P<ext>\.m[ab])$", re.I)
    SUCCESS, FAIL, UNKNOWN = "+ ", "*** ", "\t? "

    ABC_ARGS = ("-framerange {} {} -renderableonly -selection -uvwrite -writeuvsets "
                "-writevisibility -autosubd -dataformat ogawa -file {}")

    def __init__(self, order_path):
        print("[maya2abc] Process...")

        self.time0 = time()

        self.fileInfo = None
        self.packageFolder = None
        self.framePath = None

        self.logPath = None
        self.log = []
        self.succeeded = True

        self.startFrame = None
        self.endFrame = None

        self.setup(order_path)
        self.export()

    def setup(self, order_path):
        if not os.path.isfile(order_path):
            sys.exit(os.EX_OSERR)

        for plugin in ["AbcImport", "AbcExport"]:
            if not cmds.pluginInfo(plugin, q=True, loaded=True):
                if cmds.loadPlugin(plugin):
                    print("[maya2abc] plugin loaded : {}".format(plugin))
                else:
                    cmds.error("{} cannot be loaded.".format(plugin))
                    sys.exit()

        with open(order_path, 'r') as argv_f:
            order_data = json.load(argv_f)

        kwargs = order_data["kwargs"]
        self.fileInfo = kwargs["file_data"]
        self.packageFolder = package_folder = kwargs["package_folder"]
        self.framePath = kwargs["frame_path"]
        self.logPath = report_path = kwargs["report_path"]

        if not os.path.exists(package_folder):
            os.makedirs(package_folder)

        print("[maya2abc] order path : ".format(order_path))
        print("[maya2abc] package folder : ".format(package_folder))
        print("[maya2abc] report path : ".format(report_path))

    def export(self):
        print("[maya2abc] Preparing Export...")

        for path in [self.framePath, self.packageFolder]:
            if self.RE_SHOW.match(path):
                self.report("invalid path : {}".format(path), self.FAIL)
                return self.exit()

        for source_path, node_info in self.fileInfo.items():
            self.report(source_path)

            package_path = self.prepare_packaging(source_path)
            if not package_path:
                continue

            cmds.file(package_path, open=True, force=True, loadAllReferences=True)

            valid_nodes = self.find_nodes(node_info)
            if not valid_nodes:
                self.report("-", self.UNKNOWN)
                continue
            self.promote_lod(valid_nodes)
            self.validate_time_range()

            self.export_abc(valid_nodes, package_path)
            self.cleanup(package_path)

        self.export_frame_range()
        self.exit()

    def prepare_packaging(self, source_path):
        if not os.path.exists(source_path):
            self.report("{} not exists.".format(source_path), self.FAIL)
            return self.exit()
        filename = os.path.basename(source_path)
        package_path = os.path.join(self.packageFolder, filename)
        if not os.path.exists(package_path):
            print("[maya2abc] copying {} -> {}".format(source_path, package_path))
            shutil.copy2(source_path, package_path)

        return package_path

    def find_nodes(self, node_info):
        print("[maya2abc] processing nodes...")
        valid_nodes = []
        for node in node_info["node"] or cmds.ls(assemblies=True, visible=True):
            if self.promote_ref(node):
                valid_nodes.append(node)
        for namespace in node_info["namespace"]:
            if cmds.namespace(exists=namespace):
                nodes = cmds.namespaceInfo(namespace, listOnlyDependencyNodes=True)
                for node in cmds.ls(nodes, assemblies=True):
                    if self.promote_ref(node):
                        valid_nodes.append(node)
            else:
                self.report("namespace {} not exists".format(namespace), self.FAIL)

        return valid_nodes

    def promote_ref(self, node):
        selection = cmds.ls(node)
        if not selection:
            self.report("{}: not exists".format(node), self.FAIL)
            return False
        if len(selection) > 1:
            self.report("{}: duplicate(s) exist(s)".format(node), self.FAIL)
            return False

        if cmds.referenceQuery(node, isNodeReferenced=True):
            ref_file = cmds.referenceQuery(node, filename=True)
            ref_node = cmds.referenceQuery(node, referenceNode=True)
            m = self.RE_LOW_MB.search(ref_file)
            if m:
                high_file = self.RE_LOW_MB.sub(m.group("ext"), ref_file)
                if os.path.exists(high_file):
                    print("[maya2abc] {} -> {}".format(ref_file, high_file))
                    cmds.file(high_file, loadReference=ref_node)
                else:
                    self.report("{}: high res rig not exists".format(ref_node), self.FAIL)
                    return False
        return True

    def promote_lod(self, valid_nodes):
        print("[maya2abc] promoting lod types...")

        for conf in cmds.ls("variant_CONF", recursive=True, long=True):
            if conf.split('|')[1] not in valid_nodes:
                continue
            if cmds.attributeQuery("LOD_type", node=conf, exists=True):
                cmds.cutKey(conf, time=(), attribute="LOD_type", option="keys")
                cmds.setAttr("{}.LOD_type".format(conf), 0)

    def validate_time_range(self):
        print("[maya2abc] validating time range...")

        start_frame = cmds.playbackOptions(q=True, min=True)
        if self.startFrame is not None and start_frame != self.startFrame:
            self.report("start frame not match : {} != {}".format(self.startFrame, start_frame), self.FAIL)
        self.startFrame = start_frame

        end_frame = cmds.playbackOptions(q=True, max=True)
        if self.endFrame is not None and end_frame != self.endFrame:
            self.report("end frame not match : {} != {}".format(self.endFrame, end_frame), self.FAIL)
        self.endFrame = end_frame

    def export_abc(self, nodes, path):
        abc_path = self.RE_MB.sub(".abc", path)
        cmds.select(cmds.ls(nodes, dag=True))
        cmds.AbcExport(j=self.ABC_ARGS.format(self.startFrame, self.endFrame, abc_path))
        self.report("{} : exported".format(abc_path))

    def cleanup(self, path):
        if not self.RE_SHOW.match(path):
            try:
                os.remove(path)
            except OSError:
                self.report("{} is not deleted.".format(path), self.FAIL)
            else:
                folder = os.path.dirname(path)
                self.remove_empty_folders(folder)

        cmds.file(new=True, force=True)

    def remove_empty_folders(self, path):
        if not os.listdir(path):
            os.rmdir(path)
            self.remove_empty_folders(os.path.dirname(path))

    def export_frame_range(self):
        if self.startFrame is None:
            self.report("No Result", self.FAIL)
            return self.exit()

        cmds.playbackOptions(animationStartTime=self.startFrame, min=self.startFrame,
                             animationEndTime=self.endFrame, max=self.endFrame)
        cmds.file(rename=self.framePath)
        cmds.file(s=True, f=True)
        self.report("{} : saved".format(self.framePath))

        # to save frame range to a json
        json_path = self.RE_MB.sub(".json", self.framePath)
        with open(json_path, 'w') as f:
            json.dump({"startFrame": self.startFrame, "endFrame": self.endFrame}, f)
        self.report("{} : saved".format(json_path))

    def report(self, message, state=''):
        print("[maya2abc] {}".format(message))
        self.log.append("{}{}".format(state, message))
        if state == self.FAIL:
            self.succeeded = False

    def exit(self):
        m, s = divmod(time() - self.time0, 60)
        collapsed_time = "{} min. {:.0f} sec.".format(int(m), s)
        self.report("Processed for {}".format(collapsed_time))
        self.report("{}.".format("Succeeded" if self.succeeded else "Failed"))

        if self.log and self.logPath:
            with open(self.logPath, 'w') as f:
                f.writelines(self.log)
        sys.exit()


if __name__ == '__main__':
    ExportAbc(sys.argv[1])
