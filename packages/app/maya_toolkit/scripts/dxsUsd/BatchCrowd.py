
import os, sys


#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import batchCommon
    optparser = batchCommon.crowdOptParserSetup()
    opts, args= optparser.parse_args(sys.argv)
    if not opts.srcfile:
        os._exit(0)

    from pymel.all import *
    plugins = [
        'backstageMenu', 'pxrUsd', 'pxrUsdTranslators', 'AbcExport']
#        'MiarmyProForMaya%s' % os.getenv('MAYA_VER'), 'MiarmyForDexter'
#    ]
    batchCommon.InitPlugins(plugins)

    import Crowd

    if opts.onlybake:
        # opts.srcfile - crowd version dir
        Crowd.UsdSkelBakeOnly(opts.srcfile).doIt()
    else:
        if opts.meshdrive:
            import Miarmy
            Miarmy.MiarmyBatchMeshDriveExport(opts.srcfile)
        else:
            if not opts.outdir:
                os._exit(0)
            outDir = os.path.dirname(opts.outdir)
            ver    = os.path.basename(opts.outdir)
            Crowd.CrowdShotExport(expfr=opts.exportFrameRange, task=opts.task, sceneFile=opts.srcfile, fr=opts.frameRange, outDir=outDir, version=ver).doIt()

    os._exit(0)
