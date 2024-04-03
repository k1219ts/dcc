import os
import re

class TextureCtrl():
    def __init__(self, Path):
        self.texturePath = os.path.join(Path, 'texture')
        self.devTextureVersionList = []
        self.pubTextuerVersionList = []
        self.devPubTextureList = []
        self.texChannelList = []
        
    def getTexturePath(self):
        return self.texturePath
        
    def loadTextureVersionList(self, key = "pub"):
        texturePathData = os.path.join(self.texturePath, '{0}/tex'.format(key))
        textureVersionList = []
        
        if os.path.isdir(texturePathData):
            if os.listdir(texturePathData) >= 1:
                
                for listdir in os.listdir(texturePathData):
                    if os.path.isdir(os.path.join(texturePathData, listdir)):
                        textureVersionList.append(listdir)

                textureVersionList.sort(reverse=True)
                
        return textureVersionList
    
    def loadTextureChannel(self, version, key = "pub"):
        textureChannelPath = os.path.join(self.getTexturePath(), "{0}/tex/{1}".format(key, version))
        
        checkVariation = re.compile(r"\d+")
        
        if os.path.isdir(textureChannelPath):
            for listdir in os.listdir(textureChannelPath):
                removeExt = listdir.split('.')[0]
                splitRemoveExt = removeExt.split('_')
                channelIndex = -1
                if checkVariation.match(splitRemoveExt[channelIndex]):
                    channelIndex = -2
                channelName = splitRemoveExt[channelIndex]
                self.texChannelList.append(channelName)
                
        self.texChannelList = list(set(self.texChannelList))
    
    def loadVersionInfo(self):
        self.pubTextureVersionList = []
        self.pubTextureVersionList = self.loadTextureVersionList()