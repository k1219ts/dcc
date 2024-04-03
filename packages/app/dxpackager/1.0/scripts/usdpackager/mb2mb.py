# coding:utf-8
import sys
import os
import shutil
from time import time

import json
import re

from pxr import Sdf

import maya.standalone
import maya.cmds as cmds

import dbutils

class ExportMb:
    OPTIONS = ";shadingMode=none;readAnimData=1;assemblyRep=Import;importInstances=1;"
    RE_MB = re.compile(r"\.m[ab]$", re.I)
    RE_SHOW = re.compile("/(show|mach)/")
    RE_LOD_MB = re.compile(r"(_(?P<lod>(low|mid)))?(?P<ext>\.m[ab])$", re.I)
    SUCCESS, FAIL, KNOWN, UNKNOWN = "+ ", "*** ", "\t! ", "\t? "
    # NULL_MB = "/show/pipe/works/CSP/taewan.kim/packaging/null.mb"

    ABC_ARGS = ("-framerange {} {} -renderableonly -selection -uvwrite -writeuvsets "
                "-writevisibility -autosubd -dataformat ogawa -file {}")

    def __init__(self, order_path):
        print("[mb2mb] Processing...")

        self.time0 = time()

        self.startFrame = None
        self.endFrame = None
        self.frameRate = None
        self.lod = None
        self.TempRemove = None
        self.SmartBake = None
        self.packageRoot = ''
        self.packageFolder = ''
        self.packageScene = ''
        self.jobs = None
        self.pathToRoot = ''

        self.logPath = ''
        self.log = []
        self.succeeded = True

        self.setup(order_path)
        self.do_jobs()

    def setup(self, order_path):
        if not os.path.isfile(order_path):
            sys.exit(os.EX_OSERR)

        with open(order_path, 'r') as argv_f:
            order_data = json.load(argv_f)

        self.startFrame = order_data["start_frame"]
        self.endFrame = order_data["end_frame"]
        self.frameRate = order_data["fps"]

        self.shotPath = os.path.dirname(order_data["shot_path"])+'/ani/ani.usd'
        self.logPath = log_path = order_data["log_path"]
        self.lod = order_data["lod"]
        self.TempRemove = order_data["TempRemove"]
        self.SmartBake = order_data["SmartBake"]
        self.packageRoot = root = order_data["package_root"]
        self.packageFolder = package_folder = order_data["package_folder"]
        self.packageScene = order_data["package_scene"]
        self.jobs = order_data["jobs"]

        relative_path = package_folder.replace(root, '')
        num_parents = relative_path.count('/')
        self.pathToRoot = '/'.join([".."] * num_parents)

        print("[mb2mb] order path : {}".format(order_path))
        print("[mb2mb] package folder : {}".format(package_folder))
        print("[mb2mb] report path : {}".format(log_path))

        for plugin in ["pxrUsd", "pxrUsdTranslators"]:
            if not cmds.pluginInfo(plugin, q=True, loaded=True):
                if cmds.loadPlugin(plugin):
                    print("{} is loaded.".format(plugin))
                else:
                    self.report("{} is not loaded.".format(plugin), self.FAIL)
                    sys.exit()

    def do_jobs(self):
        if not os.path.exists(self.packageFolder):
            os.makedirs(self.packageFolder)

        mb_parts = []
        for source_path, order in self.jobs.items():
            self.report(source_path)
            temp_path = order["temp"]
            try:
                cmds.file(temp_path, o=True, f=True)
            except:
                self.report("{} cannot be loaded".format(temp_path), self.FAIL)
                continue
            cmds.file(rename=self.packageScene)

            nodes = self.nodes_to_export(order["tasks"])
            self.promote_lod(nodes)
            cmds.select(cmds.ls(nodes, dag=True) + nodes)
            mb_path = order["mb_path"]
            try:
                cmds.file(mb_path, exportSelected=True, type="mayaBinary", force=True,
                          preserveReferences=True, exportUnloadedReferences=True)
            except:
                self.report("{} cannot be exported".format(mb_path), self.FAIL)
            else:
                mb_parts.append(mb_path)
            self.cleanup(temp_path)

        temp_path = os.path.dirname(self.packageScene).replace('scenes', 'temp')
        if os.path.exists(temp_path):
            if self.TempRemove == True or self.TempRemove == "True":
                self.remove_folders(temp_path)
        else:
            print("[mb2mb] Temp Folders were not removed. Temp folder doesn't exist or Check TempRemove")

        dbutils.updatePackage(self.shotPath, self.packageScene, {'pkgType': 'mb'})
        self.build_scene(mb_parts)
        self.exit()

    def nodes_to_export(self, tasks):
        nodes_to_export = []
        for task_info in tasks:
            nodes = self.collect_nodes(task_info["nodes"], task_info["namespaces"])
            print("[mb2mbDebug] collect_nodes : {}".format(nodes))
            usd_nodes = self.usd_nodes(nodes)
            if usd_nodes:
                nodes_to_export += self.convert_to_refs(usd_nodes)

            cams = cmds.ls(nodes, type="dxCamera")
            if cams:
                self.bake_cams(cams)
                self.log += ["\t{}(dxCamera)".format(x) for x in cams]

            controls = self.rig_controls(cmds.ls(nodes, type="dxRig"))
            if controls:
                self.bake_rigs(controls)
            nodes_to_export += nodes

        return nodes_to_export

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

    def collect_nodes(self, nodes, namespaces):
        print("[mb2mb] collecting nodes...")
        valid_nodes = []
        for node in nodes:
            if self.promote_ref(node):
                valid_nodes.append(node)
        for namespace in namespaces:
            if cmds.namespace(exists=namespace):
                nodes = cmds.namespaceInfo(namespace, listOnlyDependencyNodes=True)
                for node in cmds.ls(nodes, assemblies=True):
                    if self.promote_ref(node):
                        valid_nodes.append(node)
            else:
                self.report("namespace {} not exists".format(namespace), self.FAIL)

        return valid_nodes

    @staticmethod
    def usd_nodes(nodes):
        assemblies = cmds.ls(nodes, type="pxrUsdReferenceAssembly")
        non_rigs = [x for x in nodes if cmds.nodeType(x) != "dxRig"]
        proxies = cmds.ls(non_rigs, dag=True, leaf=True, type="pxrUsdProxyShape")
        proxies = cmds.listRelatives(proxies, parent=True) or []
        return assemblies + proxies

    RE_D = re.compile(r".+(?=/_[23]d/)")
    RE_ASSET = re.compile(r"/_3d/asset/[^/]+")
    RE_USD = re.compile(r"(?P<name>[^/.]+)\.usd[abz]?$")

    def convert_to_refs(self, usd_nodes):
        references = []
        for source in usd_nodes:
            if not cmds.ls(source):
                continue
            path_attr = "{}.filePath".format(source)
            source_path = cmds.getAttr(path_attr)
            cmds.setAttr(path_attr, '', type="string")
            cmds.refresh()
            m = self.RE_ASSET.search(source_path)
            if not m:
                self.report("reference path error for {} : {}".format(source, source_path), self.FAIL)
                continue
            scene_folder = os.path.join(m.group(), "scenes")
            package_folder = self.RE_D.sub(self.packageRoot, os.path.dirname(source_path))
            package_folder = self.RE_ASSET.sub(scene_folder, package_folder)
            mb_name = self.mb_from_usd(source_path)
            mb_path = os.path.join(package_folder, mb_name)
            existent = self.is_existent(package_folder, mb_path)
            path = os.path.relpath(mb_path, self.packageFolder)
            nodes = cmds.file(path, force=True, reference=True, type="mayaBinary", groupReference=True,
                              loadReferenceDepth="none", returnNewNodes=True)
            if not existent:
                self.cleanup(mb_path)
            top_node = cmds.ls(nodes, transforms=True)[0]
            reference = cmds.ls(nodes, references=True)[0]
            cmds.copyAttr(source, top_node, values=True, inConnections=True, outConnections=True)
            cmds.delete(source)
            top_node = cmds.rename(top_node, source)
            references += [reference, top_node]
            self.log.append("\t{} : {}".format(source, mb_name))
        return references

    def is_existent(self, folder, path):
        existent = os.path.exists(path)
        if not existent:
            if not os.path.isdir(folder):
                os.makedirs(folder)
            # shutil.copy2(self.NULL_MB, path)
            open(path, 'w').close()
        return existent

    def cleanup(self, path):
        if not os.path.exists(path) or self.RE_SHOW.match(path):
            return
        try:
            os.remove(path)
        except OSError:
            self.report("{} is not deleted.".format(path), self.FAIL)
            return

        folder = os.path.dirname(path)
        print("[mb2mb] cleanup folder : {}".format(folder))
        self.remove_empty_folders(folder)

    def remove_empty_folders(self, path):
        if not os.listdir(path):
            os.rmdir(path)
            self.remove_empty_folders(os.path.dirname(path))

    def remove_folders(self, path):
        print("[mb2mb] remove the folders : {}".format(path))
        self.walk_remove(path)
        os.rmdir(path)

    def walk_remove(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                print("file to delete : {}".format(file_path))
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                print("folder to delete: {}".format(dir_path))
                os.rmdir(dir_path)

    def mb_from_usd(self, usd_path):
        m = self.RE_USD.search(usd_path)
        if not m:
            self.report("reference path error : {}".format(usd_path), self.FAIL)
            return
        name = m.group("name")

        layer = Sdf.Layer.FindOrOpen(usd_path)
        spec = layer.GetPrimAtPath(layer.defaultPrim)
        key, val = spec.variantSelections.items()[0]
        spec = spec.variantSets[key].variants[val].primSpec
        sub_path = spec.referenceList.prependedItems[0].assetPath
        sub_path = "{}{}".format(os.path.dirname(usd_path), sub_path[1:])

        layer = Sdf.Layer.FindOrOpen(sub_path)
        spec = layer.GetPrimAtPath(layer.defaultPrim)
        ver = spec.variantSelections.values()[0]

        return "{}_model_{}.mb".format(name, ver)

    RE_NAMESPACE = re.compile("(?P<name>.+:)[^:]+")

    def rig_controls(self, nodes):
        controls = []
        for node in nodes:
            m = self.RE_NAMESPACE.match(node)
            namespace = m.group("name") if m else ''
            controls += ["{}:{}".format(namespace, x) for x
                         in cmds.getAttr("{}.controllers".format(node))]
        return cmds.ls(controls)

    CAM_ATTRS = ["tx", "ty", "tz", "rx", "ry", "rz"]

    def bake_cams(self, cams):
        print("[mb2mbDebug] SmartBake : {}".format(self.SmartBake))
        print("[mb2mbDebug] baking cams : {}".format(cams))
        self.import_references(cams)
        nodes = cmds.ls(cams, dag=True, transforms=True)
        # cmds.bakeResults(nodes, smart=True,
        #                  simulation=True,
        #                  disableImplicitControl=True,
        #                  minimizeRotation=True,
        #                  time=(self.startFrame, self.endFrame),
        #                  preserveOutsideKeys=True,
        #                  attribute=self.CAM_ATTRS)

        ## packager layout camera smart bake fix 2023.09.06
        if self.SmartBake == True or self.SmartBake == "True":
            cmds.bakeResults(nodes, smart=True,
                            simulation=True,
                            disableImplicitControl=True,
                            minimizeRotation=True,
                            time=(self.startFrame, self.endFrame),
                            preserveOutsideKeys=True,
                            attribute=self.CAM_ATTRS)
        else:
            cmds.bakeResults(nodes, 
                            simulation=True,
                            disableImplicitControl=True,
                            minimizeRotation=True,
                            time=(self.startFrame, self.endFrame),
                            preserveOutsideKeys=True,
                            attribute=self.CAM_ATTRS)
        for node in nodes:
            self.extend_keys(node)

    def import_references(self, nodes):
        ref_nodes = cmds.ls(nodes, dag=True, transforms=True, referencedNodes=True)
        print("[mb2mbDebug] ref_nodes : {}".format(nodes))
        for reference in {cmds.referenceQuery(x, referenceNode=True) for x in ref_nodes}:
            if cmds.referenceQuery(reference, isLoaded=True):
                cmds.file(importReference=True, referenceNode=reference)

    RE_VER = re.compile(r".*/v\d{2,3}(?=/)", re.I)

    def copy_imageplanes(self, nodes):
        for node in cmds.ls(nodes, dag=True, type="camera"):
            connections = cmds.listConnections("{}.imagePlane".format(node),
                                               destination=False, shapes=True)
            if not connections:
                continue
            for connection in connections:
                image_attr = "{}.imageName".format(connection.split("->")[-1])
                path = cmds.getAttr(image_attr)
                source_folder = os.path.dirname(path)
                ver_folder = os.path.dirname(source_folder)
                destined_folder = self.RE_D.sub(self.packageRoot, source_folder)
                self.copy_folder(source_folder, destined_folder)
                new_path = self.RE_D.sub(self.packageRoot, path)
                cmds.setAttr(image_attr, new_path, type="string")

    def copy_folder(self, source_folder, destined_folder):
        num_errors = 0
        if not os.path.exists(destined_folder):
            os.makedirs(destined_folder)
        for f in os.listdir(source_folder):
            source_path = os.path.join(source_folder, f)
            destined_path = os.path.join(destined_folder, f)
            try:
                shutil.copy2(source_path, destined_path)
            except shutil.Error:
                num_errors += 1
        if num_errors:
            self.report("{} of {} not copied".format(num_errors, source_folder), self.FAIL)
        else:
            self.report("{} -> {}".format(source_folder, destined_folder), self.SUCCESS)

    RE_NUM = re.compile(r"\d")

    def extend_keys(self, node):
        attr = "{}.{{}}".format(node)
        try:
            attrs = [x for x in cmds.listAttr(node, keyable=True) or []
                     if not self.RE_NUM.search(x) and cmds.keyframe(attr.format(x), q=True)]
            if not attrs:
                return
        except:
            self.report("failed %s extend key frame" % node)
            return
        cmds.selectKey(node, attribute=attrs)
        cmds.keyTangent(inTangentType="spline", outTangentType="spline")
        cmds.setInfinity(preInfinite="linear", postInfinite="linear")
        cmds.setKeyframe(node, attribute=attrs,
                         time=(self.startFrame - 1, self.endFrame + 1))
        cmds.filterCurve()

    TIME_WARP_PLUG = "time1.enableTimewarp"

    def bake_rigs(self, nodes):
        print("[mb2mb] baking controls")
        root_layer = cmds.animLayer(q=True, root=True)
        layers = cmds.animLayer(root_layer, q=True, children=True) if root_layer else []

        end_frame = self.endFrame
        re_timed = cmds.getAttr(self.TIME_WARP_PLUG)
        if re_timed:
            cmds.currentTime(self.endFrame)
            end_frame = int(cmds.getAttr('time1.outTime')) + 1
            cmds.setAttr(self.TIME_WARP_PLUG, False)

        cmds.bakeResults(nodes, simulation=True, bakeOnOverrideLayer=bool(layers),
                         time=(self.startFrame, end_frame),
                         disableImplicitControl=False, minimizeRotation=True)

        if layers:
            for layer in [root_layer] + layers[:-1]:
                cmds.animLayer(layer, e=True, selected=False, preferred=False)
            cmds.animLayer(layers[-1], e=True, selected=True, preferred=True)

        if re_timed:
            cmds.setAttr(self.TIME_WARP_PLUG, True)

        for node in nodes:
            self.extend_keys(node)

    def promote_ref(self, node):
        print("[mb2mb] validate ref lods...")
        selection = cmds.ls(node)
        if not selection:
            self.report("{} not exists".format(node), self.FAIL)
            return False
        if len(selection) > 1:
            self.report("{} : duplicate(s)".format(node), self.FAIL)
            return False

        if cmds.referenceQuery(node, isNodeReferenced=True):
            reference = cmds.referenceQuery(node, referenceNode=True)
            ref_path = cmds.referenceQuery(node, filename=True, withoutCopyNumber=True)
            path = self.proper_path(ref_path)
            if path:
                if path != ref_path:
                    cmds.file(path, loadReference=reference)
                self.report("{} : {}".format(node, path), self.UNKNOWN)
            else:
                self.report("{} : proper ref not exists".format(reference), self.FAIL)
                return False
            dbutils.updatePackage(self.shotPath, self.packageScene, { 'pkgType': 'mb', 'file': ref_path })
        return True

    def proper_path(self, path0):
        m = self.RE_LOD_MB.search(path0)
        if m:
            lod = m.group("lod") or "high"
            if lod == self.lod:
                return path0
            ext = m.group("ext")
            if self.lod == "low":
                path = self.RE_LOD_MB.sub("_low{}".format(ext), path0)
                if os.path.isfile(path):
                    return path
            elif self.lod == "high":
                path = self.RE_LOD_MB.sub(ext, path0)
                if os.path.isfile(path):
                    return path

            path = self.RE_LOD_MB.sub("_mid{}".format(ext), path0)
            if os.path.isfile(path):
                return path

        if os.path.isfile(path0):
            return path0

    LOD_ATTR = "LOD_type"

    def promote_lod(self, valid_nodes):
        print("[mb2mb] promoting lod types...")

        for conf in cmds.ls("variant_CONF", recursive=True, long=True):
            if conf.split('|')[1] not in valid_nodes:
                continue
            if cmds.attributeQuery(self.LOD_ATTR, node=conf, exists=True):
                cmds.cutKey(conf, time=(), attribute=self.LOD_ATTR, option="keys")
                enum = cmds.attributeQuery(self.LOD_ATTR, node=conf, listEnum=True)[0].split(':')
                i = 0
                if self.lod in enum:
                    i = enum.index(self.lod)
                elif "mid" in enum:
                    i = enum.index("mid")
                cmds.setAttr("{}.LOD_type".format(conf), i)

    def build_scene(self, mb_parts):
        print("[mb2mb] rebuilding scene...")
        cmds.file(new=True, f=True)
        self.setup_time()

        for path in mb_parts:
            try:
                cmds.file(path, i=True, type="mayaBinary", force=True,
                          preserveReferences=True, loadReferenceDepth="all")
            except:
                self.report("failed to read part : {}".format(path), self.FAIL)

        print("[mb2mb] saving scene...")
        cmds.file(rename=self.packageScene)
        try:
            cmds.file(save=True, force=True)
        except:
            self.report("failed to assemble scene : {}".format(self.packageScene), self.FAIL)
        else:
            self.report(self.packageScene, self.SUCCESS)

        time_range = {"startFrame": self.startFrame, "endFrame": self.endFrame}
        json_path = self.RE_MB.sub(".json", self.packageScene)
        try:
            with open(json_path, 'w') as f:
                json.dump(time_range, f, indent=4)
        except:
            self.report("failed time range data", self.FAIL)
        else:
            self.report("time range data : {}".format(json_path), self.SUCCESS)

        for path in mb_parts:
            self.cleanup(path)

    def setup_time(self):
        frame_rate = self.frameRate
        for time_unit, fps in self.TIME_UNITS:
            if abs(frame_rate - fps) < 0.001:
                cmds.currentUnit(time=time_unit)
                break
        cmds.playbackOptions(animationStartTime=self.startFrame, min=self.startFrame,
                             animationEndTime=self.endFrame, max=self.endFrame)

    def report(self, message, state=''):
        print("[mb2mb] {}".format(message))
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
                f.write('\n'.join(self.log))

        sys.exit()

if __name__ == '__main__':
    maya.standalone.initialize()
    ExportMb(sys.argv[1])
