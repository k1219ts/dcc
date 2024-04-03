import dxMari
import mari
import os

def setupProject(projectName, assetName, geoFile):
    mari.projects.close()
    mari.projects.create(assetName, geoFile, [mari.ChannelInfo("diffC", 8192, 8192, mari.Image.DEPTH_BYTE, mari.Color(0.5, 0.5, 0.5, 0.0))])

    # resources path
    projRoot = "/show/%s_pub/asset/%s/texture" % (projectName, assetName)
    projRoot = "/show/%s/_3d/asset/%s/texture" % (projectName, assetName)
    mari.resources.setPath("MARI_DEFAULT_ARCHIVE_PATH", projRoot)
    mari.resources.setPath("MARI_DEFAULT_CAMERA_PATH", projRoot)
    mari.resources.setPath("MARI_DEFAULT_EXPORT_PATH", projRoot)
    mari.resources.setPath("MARI_DEFAULT_GEOMETRY_PATH", projRoot)
    mari.resources.setPath("MARI_DEFAULT_IMAGE_PATH", projRoot)
    mari.resources.setPath("MARI_DEFAULT_IMPORT_PATH", projRoot)

    txLayerFile = geoFile.replace(".abc", ".json")
    if os.path.exists(txLayerFile):
        txInfo = dxMari.readTxLayer_byFile(txLayerFile)
        if txInfo:
            dxMari.geoMetadata_txname_update(txInfo)

    # Output metadata
    dxMari.geoMetadata_outpath_update(projRoot)

def addObject(fn = "", version = ""):
    print(fn, version)
    if not version:
        version = "001"
    if not fn:
        print("# Error not filename")
        return
    else:
        jf = fn.replace( os.path.splitext(fn)[-1], '.json' )
        basename = os.path.basename( fn )
        geo_name = '%s_%s_Merged' % (os.path.splitext(basename)[0], version)
        mari.geo.load( fn,
                       {'name': geo_name,
                        'CreateChannels': [mari.ChannelInfo('diffC', 8192, 8192, 8),]} )

        mari.geo.setCurrent( geo_name )
        txinfo = dxMari.readTxLayer_byFile( jf )
        dxMari.geoMetadata_txname_update( txinfo )

def addVersion(fn = "", version = ""):
    print(fn, version)
    if not version:
        version = "001"
    if not fn:
        print("# Error not filename")
    else:
        geo = mari.geo.current()
        jf = fn.replace( os.path.splitext(fn[0])[-1], '.json' )

        basename = os.path.basename( fn )
        version_name = '%s_%s_Merged' % (os.path.splitext(basename)[0], version)
        geo.addVersion( fn, version_name )

        mari.geo.setCurrent(version_name)
        txinfo = dxMari.readTxLayer_byFile( jf )
        dxMari.geoMetadata_txname_update( txinfo )
