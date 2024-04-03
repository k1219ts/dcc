import os, sys
from subprocess import *
import getpass
import json
from datetime import datetime
from collections import OrderedDict
import ice
from PIL import Image
import shutil


iFLOAT = ice.constants.FLOAT

DCCROOT = os.path.dirname(os.getenv('DCCPROC'))
defaultJsonPath = '{}/packages/ext/prmantoolkit/scripts/TxMakerForKatana/resources/asset.json'.format(DCCROOT)


def linToRec709(iceImage):
    c1 = ice.Card(iFLOAT, [1.099])
    c2 = ice.Card(iFLOAT, [0.099])
    c3 = ice.Card(iFLOAT, [4.5])
    cp = ice.Card(iFLOAT, [0.45])
    t1 = iceImage.Pow(cp).Multiply(c1).Subtract(c2)
    t2 = iceImage.Multiply(c3)
    base = ice.Card(iFLOAT, [0.018])
    t3 = t1.Multiply(iceImage.Gt(base))
    t4 = t2.Multiply(iceImage.Le(base))
    result = t3.Add(t4)
    return result


class MainTxForKat():
    def __init__(self, filePath=None):
        self.filePath = filePath
        if ' ' in self.filePath:
            print('>>> Error : Check fileName')
            return
        self.fileName = os.path.splitext(os.path.basename(filePath))[0]
        # find show
        if self.filePath.find('/show/') == -1:
            print(">>> Error : Please run it in 'show'")
            return
        sptFilePath = self.filePath.split('/')
        getShow = sptFilePath[(sptFilePath.index('show') + 1)]

        envMapsPath = '/show/' + getShow + '/works/LNR/01_hdri/RenderManAssetLibrary/EnvironmentMaps'
        # search EnvironmentMaps
        if not os.path.exists(envMapsPath):
            print(">>> Error : The RenderManAssetLibrary does not exist.")
            return

        self.hdrDir = envMapsPath + '/' + self.fileName + '.rma'
        self.jsonPath = self.hdrDir + '/asset.json'
        self.exrPath = self.hdrDir + '/' + self.fileName + '.exr'
        self.acescgPath = self.hdrDir + '/' + self.fileName + '_acescg.exr'
        self.exrName = os.path.basename(self.exrPath)
        self.png50Path = self.hdrDir + '/asset_50.png'
        self.png100Path = self.hdrDir + '/asset_100.png'

        # search files
        if os.path.exists(self.exrPath):
            userInput = raw_input('Do you want do overwirte files(Y and y) : ')
            if userInput != 'Y' and userInput != 'y':
                return

        # make HDR Directory
        if not os.path.exists(self.filePath):
            print('>>> hdr file does not exist.')
            return
        if not os.path.exists(self.hdrDir):
            print('>>> Making HDR directory ...')
            os.makedirs(self.hdrDir)

        self.envtxmakeProcess(self.filePath)
        if not os.path.exists(self.exrPath):
            print('>>> exr file does not exist.')
            return
        self.jsonProcess()
        self.exrTopngProcess()


    def envtxmakeProcess(self, filePath):
        commands = ['txenvlatl', '-newer']
        commands += [filePath]
        cmd = ' '.join(commands)

        print('>>> Creating HDR ...')
        pipe = Popen(cmd, shell=True)
        pipe.wait()
        # move exr
        makedExrPath = os.path.splitext(self.filePath)[0] + '.exr'
        makedAcescgPath = os.path.splitext(self.filePath)[0] + '_acescg.exr'

        if os.path.exists(makedExrPath) and os.path.exists(makedAcescgPath):
            # copy exr
            shutil.copy(makedExrPath, self.exrPath)
            shutil.copy(makedAcescgPath, self.acescgPath)

    def jsonProcess(self):
        print('>>> Creating json ...')
        with open(defaultJsonPath, 'r') as file:
            data = json.load(file, object_pairs_hook=OrderedDict)
        # filePath
        data['RenderManAsset']['asset']['envMap']['specs']['filename'] = self.exrName
        # dependencies
        data['RenderManAsset']['asset']['envMap']['dependencies'] = [self.exrName]
        # label
        data['RenderManAsset']['label'] = os.path.splitext(self.exrName)[0]
        # resolusion
        loadImg, bg, size = self.metaData()
        data['RenderManAsset']['asset']['envMap']['specs']['displayWindowSize'] = [list(size)]
        data['RenderManAsset']['asset']['envMap']['specs']['originalSize'] = [list(size)]
        # metadata
        data['RenderManAsset']['asset']['envMap']['metadata']['source'] = self.exrPath
        strSize = str(size[0]) + ' x ' + str(size[1])
        data['RenderManAsset']['asset']['envMap']['metadata']['resolution'] = strSize
        # user
        user = getpass.getuser()
        data['RenderManAsset']['asset']['envMap']['metadata']['author'] = user
        # datatime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        data['RenderManAsset']['asset']['envMap']['metadata']['created'] = formatted_datetime
        # save
        with open(self.jsonPath, 'w') as outFile:
            json.dump(data, outFile, indent=4)


    def exrTopngProcess(self):
        pngPath = self.exrPath.replace(".exr", ".png")
        print('>>> Converting PNG ...')
        loadImg, bg, size = self.metaData()
        loadImg = linToRec709(loadImg)
        loadImg = bg.Over(loadImg)
        loadImg.Save(pngPath, ice.constants.FMT_PNG)

        # converting exrTopng
        img = Image.open(pngPath)
        # 50 png
        resized50Image = img.resize((200,200))
        resized50Image.save(self.png50Path)
        # 100 png
        resized100Image = img.resize((400,400))
        resized100Image.save(self.png100Path)
        # Delete delPngPath
        # os.remove(pngPath)
        # ffmpeg exrTopng
        # commands = 'ffmpeg -i {} -vf "scale=400:400, eq=gamma=1.5" {}'.format(self.exrPath, self.png100Path)
        # pipe = Popen(commands, shell=True)
        # pipe.wait()


    def metaData(self):
        # loading ice Metadata
        loadImg = ice.Load(self.exrPath)
        metadata = loadImg.GetMetaData()
        orgData = metadata["Original Size"].split(' ')
        size = (int(orgData[0][1:]), int(orgData[1][:-1]))
        box = [0, size[0]-1, 0, size[1]-1]
        color = [0.0, 0.0, 0.0, 1.0]
        bg = ice.FilledImage(ice.constants.FLOAT, box, color)
        return loadImg, bg, size
