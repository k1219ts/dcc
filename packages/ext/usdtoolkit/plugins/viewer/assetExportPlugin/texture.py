
from pxr import Usd, UsdGeom, UsdShade, Sdf
import glob
import os
import shutil


def copyTexture(usdviewApi,assetName,orgDir,newShow,Element,orgAssetName):
    print '\n'
    print '-------------------------     Texture Copy Start            --------------------'

    txList = []
    imageList = []
    proxyList =[]

    textureFolder = {'tex': txList,
                     'images' : imageList,
                     'proxy' : proxyList}

    treeIter = iter(Usd.PrimRange.AllPrims(usdviewApi.selectedPrims[0]))
    treeIter.next()

    for p in treeIter:
        if p.GetTypeName() == 'Mesh':
            getBasePath = p.GetAttribute('primvars:txBasePath').Get()
            getTxVer = p.GetAttribute('primvars:ri:attributes:user:txVersion').Get()
            getLayer = p.GetAttribute('primvars:txLayerName').Get()

            #find channels
            getChannels = []
            for g in p.GetPropertyNames():
                if 'userProperties:Texture:channels' in g:
                    channel = g.split(':')[3]
                    getChannels.append(channel)
            if getLayer:  # If texture name exist
                if getTxVer:
                    for i in textureFolder:
                        orgTexturePath = os.path.join(orgDir, getBasePath, i, getTxVer)
                        # print 'orgTexturePath:', orgTexturePath
                        if getChannels:
                            if os.path.exists(orgTexturePath):
                                for channel in getChannels:
                                    textureName = "{LAYER}_{CHANNEL}".format(LAYER=getLayer, CHANNEL=channel)
                                    Allchannels = glob.glob('%s/%s*' % (orgTexturePath, textureName))
                                    for a in Allchannels:
                                        if not a in textureFolder[i]:
                                            textureFolder[i].append(a)
                        else:  # Find folder
                            if os.path.exists(orgTexturePath):
                                Allchannels = glob.glob('%s/%s*' % (orgTexturePath, getLayer))
                                for a in Allchannels:
                                    if not a in textureFolder[i]:
                                        textureFolder[i].append(a)
                            else:
                                pass
                                # print "ERROR     :   texurefile could not find :", p.GetName()
                else:
                    getTxVer = []
                    for i in textureFolder:
                        orgTextureDir = os.path.join(orgDir, getBasePath, i)
                        # print p.GetName()
                        if os.path.exists(orgTextureDir):
                            if os.listdir(orgTextureDir):
                                for f in os.listdir(orgTextureDir):
                                    if "v" in f:
                                        getTxVer.append(f)
                                        last_version = getTxVer[-1]
                                        orgTexturePath = os.path.join(orgTextureDir, last_version)

                        Allchannels = glob.glob('%s/%s*' % (orgTexturePath, getLayer))
                        n=0
                        for a in Allchannels:
                            if a in textureFolder[i]:
                                pass
                            else:
                                textureFolder[i].append(a)

    print 'Get AssetInfo:   tex total:', len(txList),',image total:', len(imageList),',proxy total:', len(proxyList)


######################################################################################################################################

    # Define Info
    if Element == True:
        newAssetFolder = "{SHOWDIR}/asset/{ASSETNAME}/element/{ELEMENTNAME}".format(SHOWDIR=newShow,
                                                                                    ASSETNAME=elementAssetName,
                                                                                    ELEMENTNAME=elementName)
    else:
        newAssetFolder = "{SHOWDIR}/asset/{ASSETNAME}".format(SHOWDIR=newShow, ASSETNAME=assetName)

    # delete Asset Folder
    if newShow == '/assetlib/3D':
        if os.path.exists(newAssetFolder):
            suCmd = 'echo dexter2019 | su render -c "rm -rf %s" ' % newAssetFolder
            os.system(suCmd)
    else:
        if os.path.exists(newAssetFolder):
            cmd = "rm -rf %s" % newAssetFolder
            os.system(cmd)

    print 'Progress     :   Overwrite: Delete Asset Folder'
    print 'Progress     :   Create Asset Folder'
    #create new directory
    for i in textureFolder:
        if Element == True:
            elementAssetName = assetName.split('_')[0]
            elementName = assetName.replace('%s_' % elementAssetName, '', 1)
            newTexureDir ="{DIR}/asset/{ASSETNAME}/element/{ELEMENTNAME}/texture/{TASK}/{VER}".format(DIR= newShow,
                                                                                                      ASSETNAME= elementAssetName,
                                                                                                      ELEMENTNAME = elementName,
                                                                                                      TASK=i,
                                                                                                      VER='v001')
        else:
            newTexureDir = "{DIR}/asset/{ASSETNAME}/texture/{TASK}/{VER}".format(DIR= newShow,ASSETNAME= assetName,TASK=i,VER='v001')

        print 'Get AssetInfo:   New %sDir:' %i, newTexureDir

        #make directory
        if newShow == '/assetlib/3D':
            #create Asset Texture Folder
            if not os.path.exists(newTexureDir):
                suCmd = 'echo dexter2019 | su render -c "mkdir -p %s"' % newTexureDir
                os.system(suCmd)
            # print "Progress     :   su render : %s Folder Created" %i
        else:
            if not os.path.exists(newTexureDir):
                os.makedirs(newTexureDir)
            # print "Progress     :   %s Folder Created" %i


        for t in textureFolder[i]:
            name = os.path.basename(t)
            checkFile = os.path.join(newTexureDir, name)
            # print 'checkFile:',checkFile
            #copy
            if newShow == '/assetlib/3D':
                if os.path.exists(checkFile):
                    pass
                    # print 'exists:', checkFile
                else:
                    os.system('echo dexter2019 | su render -c "%s"' %"cp -rf %s %s/" % (t, newTexureDir))
            else:
                if os.path.exists(checkFile):
                    pass
                    # print 'exists:', checkFile
                else:
                    os.system("cp -rf %s %s/" % (t, newTexureDir))

        if i == 'tex':
            texChList = []
            baseTxDir = '/assetlib/3D/katana/image/base_tex'
            baseCh = ['diffC', 'specG', 'specR', 'norm']
            noneList = []
            for i in os.listdir(newTexureDir):
                ch = i.split('.')[0].split('_')[1]
                if not ch in texChList:
                    texChList.append(ch)
            for i in baseCh:
                if not i in texChList:
                    if not i in noneList:
                        noneList.append(i)
            if noneList:
                print 'yes:',noneList
                for n in noneList:
                    baseTxFile = baseTxDir + '/assetName_' + n + '.tex'
                    copiedFile = newTexureDir + '/assetName_' + n + '.tex'
                    newName = copiedFile.replace('assetName',orgAssetName)
                    print 'newName:',newName

                    if newShow == '/assetlib/3D':
                        os.system('echo dexter2019 | su render -c "%s"' % "cp -rf %s %s/" % (baseTxFile, newTexureDir) )
                        os.system('echo dexter2019 | su render -c "%s"' % "mv %s %s" % (copiedFile, newName))
                    else:
                        os.system("cp -rf %s %s/" % (baseTxFile, newTexureDir) )
                        os.system("mv %s %s" % (copiedFile, newName))

    print "Progress     :   Texture Copy Successed"

