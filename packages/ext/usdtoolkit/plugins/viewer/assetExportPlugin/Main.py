
import os
import texture
import model
import AddDB


def doit(usdviewApi,orgAssetName,newShow,newAssetName,Element,overwrite,tag):
    if not newAssetName == '':
        assetName = newAssetName
    else:
        assetName = orgAssetName

    print '-------------------------     New Asset Infomation     -------------------------'
    print 'Get AssetInfo:   NewShow      :',newShow
    print 'Get AssetInfo:   New AssetName:', assetName
    # print 'Get AssetInfo:   Original AssetName:', orgAssetName
    print 'Get AssetInfo:   Element      :',Element
    print 'Get AssetInfo:   Overwrite    :',overwrite
    print 'Get AssetInfo:   Tag          :', tag

    if Element == True:
        elementAssetName = assetName.split('_')[0]
        elementName = assetName.replace('%s_' % elementAssetName, '', 1)
        newAssetDir = "{SHOWDIR}/asset/{ASSETNAME}/element/{ELEMENTNAME}".format(SHOWDIR=newShow,
                                                                                 ASSETNAME=elementAssetName,
                                                                                 ELEMENTNAME=elementName)  # '/assetlib/3D/asset/apple'
        print 'Get AssetInfo:   Element New AssetDir:', newAssetDir
    else:
        newAssetDir = "{SHOWDIR}/asset/{ASSETNAME}".format(SHOWDIR=newShow,
                                                       ASSETNAME=assetName)
        print 'Get AssetInfo:   New AssetDir :', newAssetDir


    #API Start----------------------------------------------------------------------------------------------------------
    primName = usdviewApi.selectedPrims[0].GetTypeName()
    print '\n'
    print '-------------------------     Original Asset Infomation     --------------------'
    if primName == 'Xform':
        if primName == "render":
            print 'Please select Asset'

        elif primName == "zenn":
            print 'Please select Asset'

        elif primName == "proxy":
            print 'Please select Asset'

        elif primName == "Materials":
            print 'Please select Asset'

        else:
            usdPath = usdviewApi.stageIdentifier
            print 'Get AssetInfo:   usdPath      :', usdPath

            if 'song' in usdPath:
                orgDir = os.path.join('/', usdPath.split('/')[1], usdPath.split('/')[2], usdPath.split('/')[3])

            else:
                orgDir = os.path.join('/', usdPath.split('/')[1], usdPath.split('/')[2])

            print 'Get AssetInfo:   orgDir       :',orgDir
            print 'Get AssetInfo:   orgAssetName : ',orgAssetName

            if 'element' in usdPath:
                print "orginal Asset is element"
                orgMDir = "{SHOWDIR}/asset/{ASSETNAME}/element/{ELEMENTNAME}/{TASK}".format(SHOWDIR=orgDir,
                                                                                            ASSETNAME=usdPath.split('/')[-4],
                                                                                            ELEMENTNAME=orgAssetName,
                                                                                            TASK='model')

            else:
                orgMDir = "{SHOWDIR}/asset/{ASSETNAME}/{TASK}".format(SHOWDIR = orgDir,
                                                                      ASSETNAME = orgAssetName,
                                                                      TASK = 'model')
            print 'Get AssetInfo:   org modelDir :',orgMDir
            print '\n'

            if overwrite ==False:
                if os.path.exists(newAssetDir):
                    print "Exist: Please Set other Name!", newAssetDir
                    pass
                else:
                    texture.copyTexture(usdviewApi, assetName, orgDir, newShow, Element, orgAssetName)
                    model.modelExport(usdviewApi,orgDir,newShow,orgAssetName,assetName,Element,overwrite,orgMDir)

            elif overwrite == True:
                texture.copyTexture(usdviewApi, assetName, orgDir, newShow, Element, orgAssetName)
                model.modelExport(usdviewApi,orgDir,newShow,orgAssetName,assetName,Element,overwrite,orgMDir)
            else:
                pass

    else:
        print 'Please select Xform Type.'

    #Add DB
    if newShow == '/assetlib/3D':
        if Element == True:
            newUsdPath = "{SHOWDIR}/asset/{ASSETNAME}/element/{ELEMENTNAME}/{ELEMENTNAME}.usd".format(SHOWDIR=newShow,
                                                                                     ASSETNAME=elementAssetName,
                                                                                     ELEMENTNAME=elementName)
        else:
            newUsdPath = "{SHOWDIR}/asset/{ASSETNAME}/{ASSETNAME}.usd".format(SHOWDIR=newShow,
                                                                                     ASSETNAME=assetName)  # '/assetlib/3D/asset/apple'
        print 'newUsdPath:',newUsdPath
        tagName = tag
        filepath = newUsdPath
        AddDB.AddItem(filepath)
        if tag == '':
            print 'Result:   Tag is None.'
            pass
        else:
            AddDB.AddTag(assetName,tagName)
            print 'Result:   DB upload finnished.'
    else:
        pass