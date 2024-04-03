# coding: utf-8
import os, sys, re, shutil, csv, datetime, subprocess, json, yaml

from pxr import Sdf, Usd

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg
import DXUSD.Compositor as cmp

from assetconvert import AssetConvertPack
import sympackage
import dbutils

scriptsDir = os.path.dirname(__file__)

class AssetPack():
    def __init__(self, projectCode, assetType, taskList, assetList, packageDir, vendorCode, packageFmt):
        self.pkgreqDir = '/stuff/pipe/stuff/pkgreq'
        self.packageDir = packageDir
        self.logsDir = self.packageDir+'/logs'
        try: os.makedirs(self.logsDir)
        except: pass
        if not os.path.isdir(self.logsDir):
            self.logsDir = packageDir
        self.projectCode = projectCode.lower()
        self.vendorCode = vendorCode.lower()
        self.packageFmt = packageFmt.lower()
        self.assetType = assetType.lower()
        self.taskList = taskList
        self.projectDir = '/show/'+self.projectCode
        self.findDir = '%s/%s/asset'%(self.projectDir, self.assetType)
        self.foundList = []
        self.assetList = assetList
        self.primpathlist = []
        self.refprims = []
        self.sprims = []
        self.ptprims = []
        self.exprims = []
        self.geomprim = []
        self.psPathList = []
        self.resultList = []

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
        
    def walkAssetPrims(self, prim, primPath, excludePaths=''):
        try: childPrimList = prim.GetAllChildren()
        except: return

        for p in childPrimList:
            try:
                primStks = p.GetPrimStack()
                for stk in primStks:
                    if self.psPathList.count(stk.layer.realPath) < 1:
                        self.psPathList.append(stk.layer.realPath)
                        if stk.path.pathString.startswith('/_inst_src/'):
                            stklyr = utl.AsLayer(stk.layer.realPath)
                            rootSpec = utl.GetPrimSpec(stklyr, '/_inst_src', specifier='class')
                            for name in rootSpec.nameChildren.keys():
                                srcSpec = utl.GetPrimSpec(stklyr, rootSpec.path.AppendChild(name))
                                overSpec = utl.GetPrimSpec(stklyr, srcSpec.path.AppendChild('source'), specifier='over')
                                
                                try: refList = overSpec.referenceList.prependedItems
                                except: continue
                                
                                for ref in refList:
                                    relPath = os.path.join(os.path.dirname(stk.layer.realPath), ref.assetPath)
                                    absPath = os.path.abspath(relPath)
                                    absDir = os.path.dirname(absPath)
                                    if os.path.isfile(absPath):
                                        self.walkAssetDirs(self.foundList, absDir)
            except: pass

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
                        self.walkAssetPrims(p, path, excludePaths)

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
                            self.walkAssetPrims(p, path, excludePaths)

    def pkgUsdRefs(self, usdPath):        
        stage = Usd.Stage.Open(usdPath)
        dPrim = stage.GetDefaultPrim()
        dPrimPath = dPrim.GetPath().pathString
        self.walkAssetPrims(dPrim, dPrimPath)

    def walkAssetDirs(self, foundList, tgtDir, cutDir='', maxDepth=100, depth=0):
        try: lsd = os.listdir(tgtDir)
        except: return
        rlsd = sorted(lsd, reverse=True)
        latestVer = ''
        latestPrefix = ''
        for fn in rlsd:
            # if bool(re.match(r'.*[^a-zA-Z0-9]?v[0-9]{3}[^a-zA-Z0-9]?', fn)):
            #     try: prefix = re.match(r'(.*)v[0-9]{3}.*', fn).groups()[0]
            #     except: prefix = ''
            #     try: ver = re.match(r'(.*v[0-9]{3}).*', fn).groups()[0]
            #     except: ver = ''

            #     if latestPrefix != prefix:                
            #         latestVer = ''
            #         latestPrefix = prefix

            #     if latestVer == '':
            #         latestVer = ver
            #     else:
            #         if latestVer != ver:
            #             continue
                
            path = os.path.join(tgtDir, fn)
            if os.path.isdir(path):
                if maxDepth > depth:
                    self.walkAssetDirs(foundList, path, cutDir, maxDepth, depth)

            else:
                if cutDir != '':
                    path = '%s,%s'%(path, path.split('/%s/'%cutDir)[-1])
                
                foundList.append(path)

                # if fn.endswith('.usd') and fn.count('geom') < 1 and fn.count('/branch/') < 1:
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

    def packMb(self):
        # usdpackager.py에서 assetconvert로 대체
        return

    # mb, abc 심볼릭은 이 함수에서 같이 처리
    def packSym(self, fmt):
        for asset in self.assetList:
            assetDir = '%s/%s'%(self.findDir, asset)
            if fmt == 'abc':
                for task in self.taskList:
                    taskDir = '%s/%s'%(assetDir, task)
                    try: lsd = os.listdir(taskDir)
                    except: continue
                    taskFn = ''
                    rlsd = sorted(lsd, reverse=True)
                    for fn in rlsd:
                        taskVerDir = taskDir+'/'+fn
                        if bool(re.match(r'^v[0-9]{3}', fn)) and os.path.isdir(taskVerDir):
                            taskFn = asset+'_'+task+'.abc'
                            taskFilePath = taskVerDir+'/'+taskFn
                            if os.path.isfile(taskFilePath):
                                self.foundList.append(taskFilePath)
                            break

            elif fmt == 'mb':
                for task in self.taskList:
                    taskScenesDir = '%s/%s/scenes'%(assetDir, task)
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

                    texImgDir = assetDir+'/texture/images'
                    try: lstxd = os.listdir(texImgDir)
                    except: continue
                    rlstxd = sorted(lstxd, reverse=True)
                    for tfn in rlstxd:
                        texVerDir = texImgDir+'/'+tfn
                        if bool(re.match(r'^v[0-9]{3}', tfn)) and os.path.isdir(texVerDir):
                            try: lstxf = os.listdir(texVerDir)
                            except: continue
                            for txf in lstxf:
                                self.foundList.append(texVerDir+'/'+txf)
                            break

        sympackage.symWalked(self.packageDir, self.foundList)

    def packUsd(self):
        if self.assetType == '_3d':
            self.taskList.append('branch')
            self.taskList.append('material')
            self.taskList.append('texture')

        for asset in self.assetList:
            seq = asset.split('_')[0]
            assetDir = '%s/%s'%(self.findDir, asset)
            assetUsd = '%s/%s/%s.usd'%(self.findDir, asset, asset)
            # if os.path.isfile(assetUsd):
            #     self.pkgUsdRefs(assetUsd)
            for task in self.taskList:
                taskDir = '%s/%s'%(assetDir, task)
                if os.path.isdir(taskDir):
                    self.walkAssetDirs(self.foundList, taskDir)

                    taskUsdPath = taskDir+'/'+task+'.usd'
                    if os.path.isfile(taskUsdPath):
                        self.pkgUsdRefs(taskUsdPath)

                self.walkAssetDirs(self.foundList, taskDir, maxDepth=0)
            self.walkAssetDirs(self.foundList, assetDir, maxDepth=0)

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
            nowStr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            if len(self.assetList) == 1:
                nowStr = self.assetList[0]+'_'+nowStr
            logFile = self.logsDir+'/'+self.assetType+'_asset_'+nowStr+'.log'
            with open(logFile, 'w') as f:
                yaml.safe_dump(self.resultList, f, encoding='utf-8', allow_unicode=True, default_flow_style=False)

        except:
            print ('Write log error', self.assetType, self.assetList, self.taskList)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        exit(1)
    assetPack = AssetPack(sys.argv[1], sys.argv[2], [sys.argv[3]], [sys.argv[4]], sys.argv[5], sys.argv[6], sys.argv[7])
    assetPack.startPackage()