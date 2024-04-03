import getpass
import string
import maya.cmds as cmds
import maya.mel as mel
import dexcmd.alembicBatchExport as alembicBatchExport
import dexcmd.batchCommon as batchCommon
import dexcmd.previewRender as previewRender

# import saveAndPub_dev.alembicBatchExportCustom as alembicBatchExportCustom
# reload(alembicBatchExportCustom)
def localPreviewRender(logFile, Stamp, revision):
    options = previewRender.readLogfile(logFile)
    options['renderer'] = 1
    options['size'] = (1920, 1080)
    if Stamp:
        options['stamp'] = Stamp
    if revision:
        options['revision'] = True

    mel.eval('print "#---------------------------------------------------------------#\\n"')
    mel.eval('print "#\\n"')
    mel.eval('print "#		Preview Hardware 2.0 Render\\n"')
    mel.eval('print "#\\n"')
    mel.eval('print "#---------------------------------------------------------------#\\n"')

    previewRender.JsonRender(options)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# fileName = OPN_0820_ani_v01.mb
def tractorSpool(
        file=None, outPath=None, camPath=None, absPath=True,
        frameRange=None, step=1.0, just=False,
        mesh=False, meshType='render,mid,low,sim',
        node=None,
        camera=False, preview=False, stamp=False,
        layout=False, zenn=False, nextHairTask=False,
        host='tractor', separate=False, revision=False,
        offLogWrite=True, logfile=None, dbinsert=False, cleanup=True):
    # selected
    if not node:
        selection = cmds.ls(sl=True, type='dxRig')
        if selection:
            node = string.join(selection, ',')

    cmds.file(save=True)

    options = {
        'file': file, 'outPath': outPath, 'camPath': camPath, 'absPath': absPath,
        'frameRange': frameRange, 'step': step, 'just': just,
        'mesh': mesh, 'meshType': meshType, 'node': node,
        'camera': camera, 'preview': preview, 'stamp': stamp,
        'layout': layout, 'zenn': zenn, 'nextHairTask': nextHairTask,
        'host': host, 'separate': separate, 'revision': revision,
        'offLogWrite': offLogWrite, 'logfile': logfile, 'username': getpass.getuser(),
        'dbinsert': dbinsert, 'cleanup': cleanup}

    opts = Struct(**options)
    # exportClass = alembicBatchExportCustom.ExportCustom(opts)
    exportClass = alembicBatchExport.Export(opts)
    if opts.host == 'tractor':
        exportClass.spool()
    else:
        exportClass.doIt()
        if not opts.offLogWrite:
            exportClass.logWrite()
        # local preview
        if opts.preview and exportClass.m_logfile:
            localPreviewRender(exportClass.m_logfile, opts.stamp, opts.revision)

        # next task zenn cache export
        if opts.nextHairTask:
            batchCommon.nextTask_zennCache_export(exportClass.m_logfile, opts.host)
