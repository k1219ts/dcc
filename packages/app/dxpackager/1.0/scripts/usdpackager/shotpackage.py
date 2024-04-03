# coding: utf-8
import os, sys, re, shutil, csv, subprocess, json, yaml
from datetime import datetime
from distutils.dir_util import copy_tree
from collections import OrderedDict, defaultdict

from pxr import Sdf, Usd

import DXRulebook.Interface as rb
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg
import DXUSD.Compositor as cmp

import assetpackage, sympackage
import dbutils

scriptsDir = os.path.dirname(__file__)


def setShowConfig(show):
    showRbPath = '/show/{SHOW}/_config/DXRulebook.yaml'.format(SHOW=show)

    if os.path.exists(showRbPath):
        print ('>> showRbPath:{}'.format(showRbPath))
        os.environ['DXRULEBOOKFILE'] = showRbPath
    else:
        if os.environ.has_key('DXRULEBOOKFILE'):
            del os.environ['DXRULEBOOKFILE']

    rb.Reload()


class ShotPack():
    def __init__(self, projectCode, shotType, taskList, shotList, packageDir, vendorCode, packageFmt, withAsset, TempRemove, SmartBake):
        self.packageDir = packageDir
        self.logDir = os.path.join(self.packageDir, 'logs')
        if not os.path.exists(self.logDir):
            try:    os.makedirs(self.logDir)
            except: print('#### makedir ERROR:', self.logDir)
        if not os.path.isdir(self.logDir):
            self.logDir = packageDir
        self.commandDir = os.path.join(self.packageDir, "commands")
        if not os.path.exists(self.commandDir):
            try:    os.makedirs(self.commandDir)
            except: print('#### makedir ERROR:', self.commandDir)
        if not os.path.isdir(self.commandDir):
            self.commandDir = packageDir
        self.projectCode = projectCode.lower()
        self.vendorCode = vendorCode.lower()
        self.packageFmt = packageFmt.lower()
        self.shotType = shotType.lower()
        self.taskList = taskList
        self.projectDir = '/show/'+self.projectCode
        self.findDir = '%s/%s/shot'%(self.projectDir, self.shotType)
        self.foundList = []
        self.shotList = shotList
        self.withAsset = withAsset == 'True'
        self.TempRemove = TempRemove
        self.SmartBake = SmartBake

        self.primpathlist = []
        self.refprims = []
        self.sprims = []
        self.ptprims = []
        self.exprims = []
        self.geomprim = []
        self.psPathList = []
        self.assetInfo = dict()
        self.resultList = []

        # set showConfig
        setShowConfig(projectCode)

    def checkPrimExclude(self, excludePaths, path, primPath):
        # print('path:',path)
        exclude = False

        if primPath:
            if not primPath in path:
                exclude = True

        if excludePaths:
            for ex in excludePaths:
                ex = ex.lstrip()
                if ex == path or '/scatter' in path:
                    exclude = True

        # if exclude == False:
        #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>primPath:', primPath)
        #     print('path:', path)

        return exclude

    def walkShotPrims(self, prim, primPath, excludePaths=''):
        try: childPrimList = prim.GetAllChildren()
        except: return

        for p in childPrimList:
            primStks = p.GetPrimStack()
            for stk in primStks:
                try:
                    stkRefList = stk.referenceList.prependedItems
                    for strp in stkRefList:
                        stRefPath = utl.GetAbsPath(stk.layer.realPath, strp.assetPath)
                        stRefPath = os.path.abspath(os.path.join(stk.layer.realPath, strp.assetPath.replace('@', '')))
                        if self.psPathList.count(stRefPath) < 1:
                            self.psPathList.append(stRefPath)
                except: pass

                if self.psPathList.count(stk.layer.realPath) < 1:
                    self.psPathList.append(stk.layer.realPath)

                    stklyr = utl.AsLayer(stk.layer.realPath)
                    rootSpec = utl.GetPrimSpec(stklyr, '/_inst_src', specifier='class')
                    for name in rootSpec.nameChildren.keys():
                        srcSpec = utl.GetPrimSpec(stklyr, rootSpec.path.AppendChild(name))
                        overSpec = utl.GetPrimSpec(stklyr, srcSpec.path.AppendChild('source'), specifier='over')

                        try: instSrcStkRefList = overSpec.referenceList.prependedItems
                        except: continue

                        for isstrp in instSrcStkRefList:
                            istRefPath = utl.GetAbsPath(stk.layer.realPath, isstrp.assetPath)
                            istRefPath = os.path.abspath(os.path.join(stk.layer.realPath, isstrp.assetPath.replace('@', '')))
                            if self.psPathList.count(istRefPath) < 1:
                                self.psPathList.append(istRefPath)

                            try: instSrcStage = Usd.Stage.Open(istRefPath)
                            except: continue

                            instSrcPrim = instSrcStage.GetDefaultPrim()
                            instSrcPrimStakList = instSrcPrim.GetPrimStack()
                            for isstk in instSrcPrimStakList:
                                if self.psPathList.count(isstk.layer.realPath) < 1:
                                    self.psPathList.append(isstk.layer.realPath)

                            instSrcPath = instSrcPrim.GetPath().pathString
                            self.walkShotPrims(instSrcPrim, instSrcPath, excludePaths)

            path = p.GetPath().pathString
            exclude = self.checkPrimExclude(excludePaths, path, primPath)

            # if '/Cam' in path or p.GetTypeName() == 'Scope':
            if p.GetTypeName() == 'Scope':
                pass

            else:
                if p.GetTypeName() == 'PointInstancer':
                    msg.debug('[point instancing]:', path)
                    self.ptprims.append(p)
                    continue

                elif p.GetTypeName() == 'Mesh':
                    if exclude == False:
                        if not path in self.primpathlist:
                            self.primpathlist.append(path)
                    else:
                        if not path in self.exprims:
                            self.exprims.append(path)

                else:
                    if p.GetParent().GetName() == 'Layout' or p.GetParent().GetName() == 'World':
                        # self.walkShotPrims(p, path, excludePaths)
                        pass

                    else:
                        if p.HasAuthoredSpecializes():
                            if exclude == False:
                                if not path in self.primpathlist:
                                    self.primpathlist.append(path)
                                    # msg.debug('[sceneGraph instancing]:', path)
                                if not path in self.sprims:
                                    self.sprims.append(p)
                            else:
                                if not path in self.exprims:
                                    self.exprims.append(path)

                        elif p.HasAuthoredReferences():
                            if exclude == False:
                                if not path in self.primpathlist:
                                    self.primpathlist.append(path)
                                    # msg.debug('[Reference]:', path)
                                if not path in self.refprims:
                                    self.refprims.append(p)
                            else:
                                if not path in self.exprims:
                                    self.exprims.append(path)

                        elif p.GetTypeName() == 'Xform':
                            #  self.walkShotPrims(p, path, excludePaths)
                            pass

            self.walkShotPrims(p, path, excludePaths)

    def pkgUsdRefs(self, usdPath):
        stage = Usd.Stage.Open(usdPath)
        dPrim = stage.GetDefaultPrim()
        dPrimPath = dPrim.GetPath().pathString
        self.walkShotPrims(dPrim, dPrimPath)

    def walkShotDirs(self, foundList, tgtDir, cutDir='', maxDepth=100, depth=0):
        try: lsd = os.listdir(tgtDir)
        except: return
        rlsd = sorted(lsd, reverse=True)
        latestVer = ''
        latestPrefix = ''
        for fn in rlsd:
            if bool(re.match(r'.*[^a-zA-Z0-9]?v[0-9]{3}[^a-zA-Z0-9]?', fn)):
                try: prefix = re.match(r'(.*)v[0-9]{3}.*', fn).groups()[0]
                except: prefix = ''
                try: ver = re.match(r'(.*v[0-9]{3}).*', fn).groups()[0]
                except: ver = ''

                if latestPrefix != prefix:
                    latestVer = ''
                    latestPrefix = prefix

                if latestVer == '':
                    latestVer = ver
                else:
                    if latestVer != ver:
                        continue

            path = os.path.join(tgtDir, fn)
            if os.path.isdir(path):
                if maxDepth > depth:
                    self.walkShotDirs(foundList, path, cutDir, maxDepth, depth)

            else:
                if cutDir != '':
                    path = '%s,%s'%(path, path.split('/%s/'%cutDir)[-1])

                foundList.append(path)

                # if self.withAsset and fn.endswith('.usd') and fn.count('geom') < 1 and fn.count('/branch/') < 1:
                #     self.pkgUsdRefs(path)

        depth = depth + 1

    def startPackage(self):
        if '(sym)' in self.packageFmt:
            if self.packageFmt.startswith('usd'):
                self.packUsd()
            else:
                self.packSym(self.packageFmt.split('(')[0])

        else:
            if self.packageFmt.startswith('usd'):
                self.packUsd()
            elif self.packageFmt.startswith('mb'):
                self.packMb()
                return True

    def packMb(self):
        if self.shotType != '_3d':
            print ('Oops!')
            return

        method = self.packageFmt
        if method == "mb":
            self.export_mb()
        elif method == "mb(low)":
            self.export_mb("low")
        elif method == "mb(abc)":
            self.export_abc()

        log_name = [self.shotType, "shot"]
        if len(self.shotList) == 1:
            log_name.append(self.shotList[0])
        log_name += [method, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        log_path = os.path.join(self.logDir, "{}.log".format('_'.join(log_name)))
        log_text = '\n'.join(self.resultList)
        with open(log_path, 'w') as f:
            f.write(log_text)

    MAYA_REZ = ["/backstage/dcc/DCC", "rez-env",
                'maya-2018', 'dxusd_maya', 'usd_maya-19.11', 'pylibs-2']
    MB2MB_COMMAND = ["--", "mayapy", os.path.join(scriptsDir, "mb2mb.py")]
    COMMAND_TEMPLATE = {
        "command_version": "1.0",
        "commands": {}
    }
    RE_UNDER_3D = re.compile("^/show/[^/]*(?=/)")
    COPY_COMMAND = "cp"
    LOG_END = ''
    LOG_FAIL = ["Failed.", LOG_END]
    LOG_SUCCESS = ["Succeeded.", LOG_END]
    LOG_TITLE = "<{}>"
    LOG_CHECK = "* check {}"

    def export_mb(self, lod="high"):
        for shot in self.shotList:
            self.resultList.append(self.LOG_TITLE.format(shot))
            jobs = dict()

            try:
                coder = rb.Coder()
                arg = coder.N.SHOTNAME.Decode(shot)
                seq = arg.seq
            except:
                seq = shot.split('_')[0]

            shotDir = os.path.join(self.findDir, seq, shot)
            package_base = self.RE_UNDER_3D.sub(self.packageDir, shotDir)
            package_temp_folder = os.path.join(package_base, "temp")
            package_folder = os.path.join(package_base, "scenes")
            error = None
            for task, usd_path in self.task_info(shotDir, shot):
                if task == "cam":
                    self.resultList.append(task)
                    results, error = self.export_cam(usd_path, os.path.join(package_base, "cam"))
                    self.resultList += results
                    if error:
                        self.resultList += [error] + self.LOG_FAIL
                        break
                    if len(self.taskList) == 1:
                        break
                result = self.mb_info(task, usd_path)
                if isinstance(result, str):
                    error = True
                    self.resultList += [task, result] + self.LOG_FAIL
                    break
                for path, info in result.items():
                    if path in jobs:
                        jobs[path]["tasks"].append(info)
                    else:
                        filename = os.path.basename(path)
                        jobs[path] = OrderedDict([("temp", os.path.join(package_temp_folder, filename)),
                                                  ("mb_path", os.path.join(package_folder, filename)),
                                                  ("tasks", [info])])
            if error:
                continue
            if self.taskList == ["cam"]:
                self.resultList += self.LOG_SUCCESS
                continue

            self.resultList.append("SHOT")
            code = "{}_{}".format(shot, lod) if lod != "high" else shot
            shot_usd_path = os.path.join(shotDir, "{}.usd".format(shot))
            TempRemove = self.TempRemove
            SmartBake = self.SmartBake
            print("TempRemove = {} ".format(TempRemove))
            print("SmartBake = {} ".format(SmartBake))
            job_info = [("lod", lod),
                        ("TempRemove", TempRemove),
                        ("SmartBake", SmartBake),
                        ("package_folder", package_folder),
                        ("package_scene", os.path.join(package_folder, "{}.mb".format(code))),
                        ("jobs", jobs)]
            result = self.order_form(code, shot_usd_path, job_info)
            if isinstance(result, str):
                self.resultList += [result] + self.LOG_FAIL
                continue
            self.resultList.append(self.LOG_CHECK.format(result["log_path"]))
            order_path = os.path.join(self.commandDir, "{}.json".format(code))
            with open(order_path, 'w') as f:
                json.dump(result, f, indent=4)

            if not os.path.exists(package_temp_folder):
                os.makedirs(package_temp_folder)
            for path, info in jobs.items():
                subprocess.Popen([self.COPY_COMMAND, path, info["temp"]]).wait()
                print(self.COPIED.format(path, info["temp"]))

            print("[mb2mb] command : {} ".format(self.MAYA_REZ + self.MB2MB_COMMAND + [order_path]))
            subprocess.Popen(self.MAYA_REZ + self.MB2MB_COMMAND + [order_path]).wait()
            self.resultList.append(self.LOG_END)

    RE_SUB_TASK = re.compile(r"\./(?P<task>[^/]+)/")
    RE_CURRENT_FOLDER = re.compile(r"^\.(?=/)")

    def task_info(self, shot_folder, shot, task_list=None):
        task_list = task_list or self.taskList
        usd_path = os.path.join(shot_folder, "{}.usd".format(shot))
        layer = Sdf.Layer.FindOrOpen(usd_path)
        task_info = []
        for path in layer.subLayerPaths:
            print("subLayerPaths found : {}".format(path))
            task = self.RE_SUB_TASK.search(path).group("task")
            if task in task_list:
                abs_path = os.path.abspath(os.path.join(shot_folder, path))
                task_info.append([task, abs_path])
        return task_info

    RE_FILE_TASK = re.compile(r"/(?P<task>[^/]+)\.usd[az]?", re.I)
    RE_MB = re.compile(r"\.m[ab]$", re.I)
    TASK2KEY = {
        "cam": "camera",
        "ani": "geoCache",
        "layout": "layout",
        "groom": "zenn",
        "sim": "sim",
        "crowd": "crowd"
    }

    def mb_info(self, task, source_path):
        if not os.path.isfile(source_path):
            return "- no task folder : {}".format(source_path)

        mb_info = dict()
        for mb_path, namespaces in self.scene_info(source_path).items():
            if not os.path.isfile(mb_path):
                return "- invalid path : {}".format(mb_path)
            json_path = self.RE_MB.sub(".json", mb_path)
            if os.path.isfile(json_path):
                node_list, namespace_list = self.nodes_from_json(json_path, task, namespaces)
            else:
                return "{}: invalid path".format(json_path)
            # if task == "layout":
            #     unknown_nodes, missing_nodes = self.check_assets(node_list)
            #     message = ''
            #     if unknown_nodes:
            #         message = "- unknown asset(s) : {}".format(", ".join(unknown_nodes))
            #     if missing_nodes:
            #         message = "{}{}- missing asset(s) : {}".format(message,
            #                                                        '\n' if message else '',
            #                                                        ", ".join(missing_nodes))
            #     if message:
            #         return message

            mb_info[mb_path] = OrderedDict([
                ("task", task),
                ("nodes", node_list),
                ("namespaces", namespace_list),
            ])

        return mb_info

    def nodes_from_json(self, path, task, namespaces):
        with open(path, 'r') as f:
            json_data = json.load(f)
        match_set = set()
        node_list = []
        key = self.TASK2KEY[task]
        if task in ["cam", "layout"]:
            node_list = [x[0] for x in json_data[key]]
            namespace_list = []
        else:
            for data in json_data[key]:
                node_name = data[0]
                ns = rb.Coder().N.USD.ani.Decode(node_name, 'SHOT').nslyr
                if ns in namespaces:
                    node_list.append(node_name)
                    match_set.add(ns)
            namespace_list = list(set(namespaces) - match_set)

        return node_list, namespace_list

    def scene_info(self, usd_path):
        if not os.path.isfile(usd_path):
            return dict()
        layer = Sdf.Layer.FindOrOpen(usd_path)
        if "sceneFile" in layer.customLayerData:
            return {layer.customLayerData["sceneFile"]: [layer.defaultPrim]}

        scene_info = defaultdict(list)
        spec = layer.GetPrimAtPath(layer.defaultPrim)
        layer_folder = os.path.dirname(usd_path)
        paths = [os.path.abspath(os.path.join(layer_folder, x))
                 for x in self.find_references(spec) + list(layer.subLayerPaths)]
        for path in paths:
            for mb_path, namespaces in self.scene_info(path).items():
                scene_info[mb_path] += namespaces
        return scene_info

    def find_references(self, prim_spec):
        if not prim_spec:
            return []
        paths = []
        for key, val in prim_spec.variantSelections.items():
            var_spec = prim_spec.variantSets[key].variants[val].primSpec
            if var_spec.hasReferences:
                paths += [x.assetPath for x in var_spec.referenceList.prependedItems]
        for spec in prim_spec.nameChildren:
            paths += self.find_references(spec)
        return paths

    RE_D = re.compile(r".+(?=/_\dd/)")
    RE_ASSET = re.compile(r"/_3d/asset/[^/]+")
    RE_USD = re.compile(r"(?P<name>[^/.]+)\.usd[abz]?$")

    def check_assets(self, node_list):
        if not self.assetInfo:
            self.scan_asset()
        unknown_nodes = []
        missing_nodes = set()
        for node in node_list:
            if node not in self.assetInfo:
                unknown_nodes.append(node)
                continue
            asset, usd_folder = self.assetInfo[node]
            m = self.RE_ASSET.search(usd_folder)
            if not m:
                unknown_nodes.append(node)
                continue
            scene_folder = os.path.join(m.group(), "scene")
            folder = self.RE_D.sub(self.packageDir, usd_folder)
            folder = self.RE_ASSET.sub(scene_folder, folder)
            usd_path = os.path.join(self.assetInfo[node][1], "{}.usd".format(node))
            mb_path = os.path.join(folder, self.mb_from_usd(usd_path))
            if not mb_path:
                unknown_nodes.append(node)
            elif not os.path.isfile(mb_path):
                missing_nodes.add(asset)
        return unknown_nodes, list(missing_nodes)

    def scan_asset(self):
        tag = rb._RBROOT.tag
        root = os.path.join(self.projectDir, tag["PUB3"].value,  tag["ASSET"].value)
        asset_items = [(x, os.path.join(root, x)) for x in os.listdir(root)]
        asset_info = {x: (x, y) for x, y in asset_items if os.path.isdir(y)}
        for asset, path in asset_info.values():
            branch_root = os.path.join(path, "branch")
            if not os.path.isdir(branch_root):
                continue
            branch_items = [(x, os.path.join(branch_root, x)) for x in os.listdir(branch_root)]
            branch_info = {x: (asset, y) for x, y in branch_items if os.path.isdir(y)}
            asset_info.update(branch_info)
        self.assetInfo = asset_info

    def mb_from_usd(self, usd_path):
        m = self.RE_USD.search(usd_path)
        if not m:
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

    def reference_paths(self, prim):
        if prim.HasAuthoredReferences():
            references = prim.GetMetadata("references")
            return [x.assetPath for x in references.GetAddedOrExplicitItems()]

        reference_paths = []
        for sub_prim in prim.GetAllChildren():
            reference_paths += self.reference_paths(sub_prim)
        return reference_paths

    RE_CAM_VER = re.compile(r"_(?P<name>[^_]+)_camera_v(?P<ver>\d+)\.(mb|ma)", re.I)
    COPIED = "+ {}\n  -> {}"
    COPY_FAILED = "- {}\n  -> FAILED"

    def export_cam(self, usd_path, package_folder):
        task_folder = os.path.dirname(usd_path)
        layer = Sdf.Layer.FindOrOpen(usd_path)
        version_info = self.find_cam_version(layer.GetPrimAtPath(layer.defaultPrim))
        if not version_info:
            return [], "- No Cam"
        version, path = version_info

        results, error = self.copy_cam_folder(task_folder, version, package_folder)
        if error:
            return results, error

        for source_folder in self.find_imageplane_folders(usd_path):
            destined_folder = self.RE_UNDER_3D.sub(self.packageDir, source_folder)
            try:
                copy_tree(source_folder, destined_folder)
            except shutil.Error:
                return results, self.COPY_FAILED.format(source_folder)
            results.append(self.COPIED.format(source_folder, destined_folder))
        return results, None

    @staticmethod
    def find_cam_version(spec):
        nodes = []
        while spec.name != "Cam":
            nodes += list(spec.nameChildren)
            if not nodes:
                return
            spec = nodes.pop(0)
        for key, value in spec.variantSelections.items():
            if key == "camVer":
                ver_spec = spec.variantSets[key].variants[value].primSpec
                path = ver_spec.referenceList.prependedItems[0].assetPath
                return value, path

    def copy_cam_folder(self, task_folder, version, package_folder):
        results = []
        source_folder = os.path.join(task_folder, version)
        destined_folder = os.path.join(package_folder, version)
        try:
            copy_tree(source_folder, destined_folder)
        except shutil.Error:
            return results, self.COPY_FAILED.format(source_folder)
        results.append(self.COPIED.format(source_folder, destined_folder))

        scenes_folder = os.path.join(task_folder, "scenes")
        cam_scene = self.find_cam_scene(scenes_folder, version)
        if not cam_scene:
            return results, "- No Scene File"

        source_path = os.path.join(scenes_folder, cam_scene)
        destined_path = os.path.join(package_folder, cam_scene)
        try:
            shutil.copy2(source_path, destined_path)
        except shutil.Error:
            return results, self.COPY_FAILED.format(source_path)
        results.append(self.COPIED.format(source_path, destined_path))
        dbutils.updatePackage(source_path, destined_path, {'pkgType': 'mb'})
        return results, None

    def find_cam_scene(self, folder, target_ver):
        scene, name, ver = '', '', ''
        for mb in os.listdir(folder):
            m = self.RE_CAM_VER.search(mb)
            if not m:
                continue
            n, v, _ = m.groups()
            if n == "None" and v == target_ver:
                return scene
            if scene:
                if v < ver or (v == ver and name == "None"):
                    continue
            scene = mb
            name, ver = n, v
        return scene

    RE_SHOW = re.compile("/show/")

    def find_imageplane_folders(self, usd_path):
        stage = Usd.Stage.Open(usd_path)
        image_planes = []
        for prim in stage.Traverse():
            if prim.GetName() == "PlateImage":
                image_planes.append(prim)
        image_paths = []
        for prim in image_planes:
            if not prim.HasAttribute("inputs:file"):
                continue
            attribute = prim.GetAttribute("inputs:file")
            samples = attribute.GetTimeSamples()
            if not samples:
                continue
            path = attribute.Get(time=samples[0]).path
            for layer in prim.GetPrimStack():
                if layer.name != "PlateImage":
                    continue
                current_path = layer.layer.realPath
                if not self.RE_SHOW.match(current_path):
                    continue
                current_folder = os.path.dirname(current_path)
                abs_path = os.path.abspath(os.path.join(current_folder, path))
                ver_path = os.path.dirname(os.path.dirname(abs_path))
                image_paths.append(ver_path)
        return image_paths

    def order_form(self, code, shot_path, job_info):
        if not os.path.isfile(shot_path):
            return "- no shot folder : {}".format(shot_path)
        layer = Sdf.Layer.FindOrOpen(shot_path)
        info = ([("command_version", "1.0"),
                 ("shot_path", shot_path),
                 ("start_frame", layer.startTimeCode),
                 ("end_frame", layer.endTimeCode),
                 ("fps", layer.framesPerSecond),
                 ("log_path", os.path.join(self.logDir, "{}.log".format(code))),
                 ("package_root", self.packageDir)]
                + job_info)
        return OrderedDict(info)

    USDCAT_COMMAND = ["/backstage/dcc/DCC", "rez-env", "usd_core-20.08",
                      "--", "usdcat", "--flatten", "--out"]
    USD2ABC_COMMAND = ["--", "mayapy", os.path.join(scriptsDir, "usd2abc.py")]

    def export_abc(self):
        for shot in self.shotList:
            self.resultList.append(self.LOG_TITLE.format(shot))

            try:
                coder = rb.Coder()
                arg = coder.N.SHOTNAME.Decode(shot)
                seq = arg.seq
            except:
                seq = shot.split('_')[0]

            shotDir = os.path.join(self.findDir, seq, shot)
            abc_info = self.abc_info(shot, shotDir)
            code = "{}_abc".format(shot)

            error = False
            _, cam_path = self.task_info(shotDir, shot, ["cam"])[0]
            for source_folder in self.find_imageplane_folders(cam_path):
                destined_folder = self.RE_UNDER_3D.sub(self.packageDir, source_folder)
                try:
                    copy_tree(source_folder, destined_folder)
                except shutil.Error:
                    error = True
                    self.resultList += [self.COPY_FAILED.format(source_folder)] + self.LOG_FAIL
                    break
                self.resultList.append(self.COPIED.format(source_folder, destined_folder))
            if error:
                continue

            usd_path = os.path.join(shotDir, "{}.usd".format(shot))
            order_form = self.order_form(code, usd_path, abc_info)
            if isinstance(order_form, str):
                self.resultList += [order_form] + self.LOG_FAIL
                continue
            order_path = os.path.join(self.commandDir, "{}.json".format(code))
            with open(order_path, 'w') as f:
                json.dump(order_form, f, indent=4)

            subprocess.Popen(self.MAYA_REZ + self.USD2ABC_COMMAND + [order_path]).wait()
            self.resultList += [self.LOG_CHECK.format(order_form["log_path"]), self.LOG_END]

    def abc_info(self, code, source_folder):
        package_root = self.RE_UNDER_3D.sub(self.packageDir, source_folder)
        package_folder = os.path.join(package_root, "scenes")
        abc_info = [("package_folder", package_folder),
                    # ("temp", os.path.join(package_folder, "{}_abc.usd".format(code))),
                    ("abc_path", os.path.join(package_folder, "{}.abc".format(code))),
                    ("frame_path", os.path.join(package_folder, "{}_abc.mb".format(code)))]
        return abc_info

    # mb, abc 심볼릭은 이 함수에서 같이 처리
    def packSym(self, fmt):
        for shot in self.shotList:

            try:
                coder = rb.Coder()
                arg = coder.N.SHOTNAME.Decode(shot)
                seq = arg.seq
            except:
                seq = shot.split('_')[0]

            shotDir = '%s/%s/%s'%(self.findDir, seq, shot)
            if fmt == 'abc':
                for task in self.taskList:
                    taskDir = '%s/%s'%(shotDir, task)
                    try: lsd = os.listdir(taskDir)
                    except: continue
                    taskFn = ''
                    rlsd = sorted(lsd, reverse=True)
                    for fn in rlsd:
                        taskVerDir = taskDir+'/'+fn
                        if bool(re.match(r'^v[0-9]{3}', fn)) and os.path.isdir(taskVerDir):
                            taskFn = shot+'_'+task+'.abc'
                            taskFilePath = taskVerDir+'/'+taskFn
                            if os.path.isfile(taskFilePath):
                                self.foundList.append(taskFilePath)
                            break

            elif fmt == 'mb':
                for task in self.taskList:
                    taskScenesDir = '%s/%s/scenes'%(shotDir, task)
                    try: lsd = os.listdir(taskScenesDir)
                    except: continue
                    taskFn = ''
                    rlsd = sorted(lsd, reverse=True)
                    for fn in rlsd:
                        if fn.endswith('.'+fmt):
                            taskFn = fn
                            break

                    taskFilePath = taskScenesDir+'/'+taskFn
                    if os.path.isfile(taskFilePath):
                        self.foundList.append(taskFilePath)

        sympackage.symWalked(self.packageDir, self.foundList)

    def packUsd(self):
        for shot in self.shotList:

            try:
                coder = rb.Coder()
                arg = coder.N.SHOTNAME.Decode(shot)
                seq = arg.seq
            except:
                seq = shot.split('_')[0]

            shotDir = '%s/%s/%s'%(self.findDir, seq, shot)
            shotUsd = '%s/%s/%s/%s.usd'%(self.findDir, seq, shot, shot)
            # if os.path.isfile(shotUsd):
            #     self.pkgUsdRefs(shotUsd)
            for task in self.taskList:
                taskDir = '%s/%s'%(shotDir, task)
                if os.path.isdir(taskDir):
                    self.walkShotDirs(self.foundList, taskDir)

                    taskUsdPath = taskDir+'/'+task+'.usd'
                    if os.path.isfile(taskUsdPath):
                        self.pkgUsdRefs(taskUsdPath)

                    if self.shotType == '_3d':
                        if task == 'cam':
                            imageplaneDir = taskDir.replace('/_3d/', '/_2d/').replace('/cam', '/imageplane')
                            if os.path.isdir(imageplaneDir):
                                self.walkShotDirs(self.foundList, imageplaneDir)

                self.walkShotDirs(self.foundList, taskDir, maxDepth=0)
            self.walkShotDirs(self.foundList, shotDir, maxDepth=0)

        if self.packageFmt == 'usd(sym)':
            sympackage.symWalked(self.packageDir, self.foundList+self.psPathList)
        else:
            self.copyWalked()

    def copyWalked(self):
        fileCount = 1
        for path in self.foundList:
            copyResult = 'failed'
            try:
                srcPath = path.split(',')[0]
                relPath = path.split(',')[-1].split('/'+self.projectCode+'/')[-1]
                targetPath = '%s/%s'%(self.packageDir, relPath)
            except:
                print (srcPath, copyResult)
                self.resultList.append('-. %s %s'%(srcPath, copyResult))
                continue

            targetDir = os.path.dirname(targetPath)
            if not os.path.isdir(targetDir):
                try: os.makedirs(targetDir)
                except: pass

            if os.path.isdir(targetDir):
                print (srcPath, targetPath),
                try:
                    if not os.path.isfile(targetPath) or os.path.getsize(srcPath) != os.path.getsize(targetPath):
                        shutil.copy2(srcPath, targetPath)
                    copyResult = 'ok'
                except: pass
                dbutils.updatePackage(srcPath, targetPath)
                print (copyResult)

            self.resultList.append({
                'num': fileCount,
                'src': srcPath,
                'dst': targetPath,
                'result': copyResult
            })
            fileCount += 1

        try:
            nowStr = datetime.now().strftime('%Y%m%d%H%M%S')
            if len(self.shotList) == 1:
                nowStr = self.shotList[0]+'_'+nowStr
            logFile = self.logDir + '/' + self.shotType + '_shot_' + nowStr + '.log'
            with open(logFile, 'w') as f:
                yaml.safe_dump(self.resultList, f, encoding='utf-8', allow_unicode=True, default_flow_style=False)

        except:
            print ('Write log error {}'.format(self.shotType, self.shotList, self.taskList))

        if len(self.psPathList) > 0:
            assetList = []
            for ps in self.psPathList:
                psPath = os.path.abspath(ps)
                psDir = psPath
                if os.path.isfile(psPath):
                    psDir = os.path.dirname(psPath)

                if self.withAsset and ps.count('/asset/') > 0:
                    assetRelPath = psPath.split('/asset/')[-1]
                    assetRelPathTok = assetRelPath.split('/')
                    if len(assetRelPathTok) > 1:
                        assetName = assetRelPathTok[0]
                        taskName = assetRelPathTok[1]
                        if not assetName in assetList:
                            print ('{} Asset package by shot'.format(assetName))
                            assetTaskList = ['model', 'rig', 'clip']
                            if 'crowd' in self.taskList:
                                assetTaskList.append('agent')
                            assetPack = assetpackage.AssetPack(self.projectCode, '_3d', assetTaskList, [assetName], self.packageDir, 'vendorCode', self.packageFmt)
                            assetPack.startPackage()
                            assetList.append(assetName)

                    else:
                        self.walkShotDirs(self.foundList, psDir)
                else:
                    self.walkShotDirs(self.foundList, psDir)

if __name__ == '__main__':
    if len(sys.argv) < 8:
        exit(1)

    print ('argv : {}'.format(sys.argv))

    shotPack = ShotPack(sys.argv[1], sys.argv[2], sys.argv[3].split('_'), [sys.argv[4]], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
    shotPack.startPackage()
