import os
import string
import re
import subprocess
import nuke
import nukescripts
import precomp
# reload(precomp)
import stamp
import json

import DXRulebook.Interface as rb
# reload(rb)
import nukeCommon as comm
# reload(comm)

def getTop(node):
    if node.input(0):
        node = getTop(node.input(0))
    else:
        print("no input")
    return node


def makeJpegWrite(show, seq, shotName, task, department, argv, colorspace, fullPath):
    renderPath = ''

    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'jpg'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    filename = coder.F.IMAGES.BASE.Encode(**argv)

    if department == 'comp':
        renderPath = os.path.join(departDir, task, 'images', argv.ext,
                                  '_'.join([task, argv.ver]))
    elif department == 'fx':
        if "/dev/" in fullPath:
            renderPath = os.path.join(departDir, 'dev', 'precomp', argv.ext,
                                      '_'.join([shotName, task, argv.ver]))
        elif '/pub/' in fullPath:
            renderPath = os.path.join(departDir, 'pub', 'precomp', argv.ext,
                                      '_'.join([shotName, task, argv.ver]))
        else:
            renderPath = os.path.join(departDir, 'precomp', argv.ext,
                                      '_'.join([shotName, task, argv.ver]))
    elif department == 'lighting':
        renderPath = os.path.join(departDir, 'precomp', argv.ext, argv.ver)

    w= nuke.createNode('Write', inpanel=True)

    w.knob('file_type').setValue("jpeg")
    w.knob('_jpeg_quality').setValue(1)

    print('colorSpace:', colorspace)

    # read show _config
    configData = comm.getDxConfig()

    if colorspace == 'sRGB':
        w.knob('colorspace').setValue('Output - Rec.709')
        w.knob('_jpeg_sub_sampling').setValue('4:4:4')
        if configData:
            if configData['colorSpace'].get('in')=='Cineon':
                w.knob('colorspace').setValue('rec709')
                w.knob('_jpeg_sub_sampling').setValue('4:4:4')
            elif configData['colorSpace'].get('in')=='rec709':
                w.knob('colorspace').setValue('rec709')
                w.knob('_jpeg_sub_sampling').setValue('4:4:4')
    elif colorspace == 'Cineon':
        renderPath += '_log'
        w.knob('colorspace').setValue('Cineon')
    elif colorspace == 'default(sRGB)':
        w.knob('colorspace').setValue('Output - sRGB')
        w.knob('_jpeg_sub_sampling').setValue('4:4:4')
        pass

    outputPath = os.path.join(renderPath, filename)
    w.knob('file').setValue(outputPath)

def makeExrWrite(show, seq, shotName, task, department, argv):
    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'exr'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    filename = coder.F.IMAGES.BASE.Encode(**argv)

    verPath = '_'.join([shotName, task,argv.ver])
    renderPath = os.path.join(departDir, task, 'images', 'exr', verPath)
    # add 'DXT'
    verPath_DXT = '_'.join([shotName, task,'DXT', argv.ver])
    renderPath_DXT = os.path.join(departDir, task, 'images', 'exr', verPath_DXT)

    ref = nuke.createNode('Reformat')
    ref['type'].setValue('to box')
    ref['box_fixed'].setValue(True)

    # exr default options
    w = nuke.createNode('Write', inpanel=True)
    w.knob('file_type').setValue("exr")
    w.knob('datatype').setValue("16 bit half")
    w.knob('compression').setValue("none")
    w.knob('metadata').setValue("all metadata")
    w['channels'].setValue('rgb')
    w['autocrop'].setValue(False)

    # read show _config
    configData = comm.getDxConfig()
    if configData:
        ref['box_width'].setValue(configData['delivery']['resolution'][0])
        ref['box_height'].setValue(configData['delivery']['resolution'][1])

        w['colorspace'].setValue(str(configData['colorSpace']['out']))

        # added exr options (from _config)
        if configData['delivery'].get('options'):
            for key, value in configData['delivery']['options'].items():
                w.knob(str(key)).setValue(str(value))

    # export path
    ShowName = os.getenv('SHOW')    # add folder name 'DXT'
    if ShowName == 'yyh':
        outputPath_DXT = os.path.join(renderPath_DXT, filename)
        w.knob('file').setValue(outputPath_DXT)
    else:
        outputPath = os.path.join(renderPath, filename)
        w.knob('file').setValue(outputPath)

    return w

def makeDpxWrite(show, seq, shotName, task, department, argv, isMask=False):
    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'dpx'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    verPath = '_'.join([shotName, task, argv.ver])

    if isMask:
        renderPath = os.path.join(departDir, task, 'mask', argv.ext, verPath)
        filename = '_'.join([shotName, task, 'mask', argv.ver]) + '.%s.%s' % (argv.frame, argv.ext)

        alphaShuffle = nuke.createNode("Shuffle")
        alphaShuffle['red'].setValue('a')
        alphaShuffle['green'].setValue('a')
        alphaShuffle['blue'].setValue('a')
        alphaShuffle['alpha'].setValue('a')

        w = nuke.createNode('Write', inpanel=True)
        w.knob('colorspace').setValue("Cineon")
        w.knob('channels').setValue('rgba')
        w.knob('file_type').setValue(argv.ext)
        w.knob('datatype').setValue("10 bit")
        w.knob('file').setValue(renderPath + '/' + filename)
    else:
        renderPath = os.path.join(departDir, task, 'images', argv.ext, verPath)
        filename = '_'.join([shotName, task, argv.ver]) + '.%s.%s' % (argv.frame, argv.ext)
        outputPath = os.path.join(renderPath, filename)

        w = nuke.nodes.Write()
        w.setInput(0, nuke.selectedNode())
        w.knob('file_type').setValue(argv.ext)
        w.knob('datatype').setValue("10 bit")
        w.knob('file').setValue(outputPath)
        nuke.show(w)

        cmValue = nuke.root().knob('colorManagement').value()
        logLutValue = nuke.root().knob('logLut').value()

        ##OCIO setting
        #if cmValue == 'OCIO':
        #    w.knob('colorspace').setValue(logLutValue)
        #else:
        #    pass
        #    #w.knob('colorspace').setValue("Cineon")

def makePrecompWrite(show, seq, shotName, task, department, argv):
    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'exr'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    renderPath = os.path.join(departDir, task, 'src/')
    filename = 'precomp_' + argv.ver + '.%04d.exr'

    precompUI = precomp.PrecompUI(None, renderPath, filename, argv.ver)
    if precompUI.exec_():
        w = nuke.nodes.Write()
        w['channels'].setValue('rgba')
        w['file_type'].setValue('exr')
        w['autocrop'].setValue(True)
        w['file'].setValue(precompUI.result)

        w.setInput(0, nuke.selectedNode())


def makePngWrite(show, seq, shotName, task, department, argv):
    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'png'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    verPath = '_'.join([shotName, task, argv.ver])
    renderPath = os.path.join(departDir, task, 'images', argv.ext, verPath)
    filename = '_'.join([shotName, task, argv.ver]) + '.%s.%s' % (argv.frame, argv.ext)
    outputPath = os.path.join(renderPath, filename)

    w = nuke.createNode('Write', inpanel=True)
    w.knob('colorspace').setValue('sRGB')

    w.knob('file_type').setValue(argv.ext)
    w.knob('channels').setValue('rgba')
    w.knob('file').setValue(outputPath)

def makeTiffWrite(show, seq, shotName, task, department, argv):
    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'tiff'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    shotVer = '_'.join([shotName,task,argv.ver])
    # maskDir = '_'.join([shotName,task,'mask1'])
    renderPath = os.path.join(departDir, task, 'tiff', shotVer)
    fileName = '_'.join([shotName, task, argv.ver])
    tiffName = fileName + '.%s.%s' % (argv.frame, argv.ext)
    filename = os.path.join(renderPath, tiffName)

    crop = nuke.createNode("Crop", inpanel=False)
    w = nuke.createNode('Write', inpanel=True)
    w.knob('file_type').setValue(argv.ext)
    w.knob('compression').setValue("none")
    w.knob('channels').setValue("rgba")
    w.knob('file').setValue(filename)

    configData = comm.getDxConfig()
    if configData:
        if 'ACES' in configData['colorSpace']:
            w['colorspace'].setValue('Output - Rec.709')
    else:
        w['colorspace'].setValue('rec709')

def makeMaskTiffWrite(show, seq, shotName, task, department, argv):
    coder = rb.Coder()
    argv.frame = '%04d'
    argv.ext = 'tiff'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    shotVer = '_'.join([shotName,task,argv.ver])
    # shotName + '_' + task + '_' + argv.ver
    maskDir = '_'.join([shotName,task,'mask1'])
    renderPath = os.path.join(departDir, task, 'mask', shotVer, maskDir)
    # renderPath = os.path.join(departDir, task, 'mask', shotVer, 'mask1')
    maskFileName = '_'.join([shotName, task, argv.ver, 'mask1.%s.%s' % (argv.frame, argv.ext)])
    filename = os.path.join(renderPath, maskFileName)

    crop = nuke.createNode("Crop", inpanel=False)
    w = nuke.createNode('Write', inpanel=True)
    w.knob('file_type').setValue(argv.ext)
    w.knob('compression').setValue("none")
    w.knob('channels').setValue("rgba")
    w.knob('file').setValue(filename)

def makeMxfWrite(show, seq, shotName, task, department, argv):
    coder = rb.Coder()
    argv.ext = 'mxf'
    departDir = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv), department)
    renderPath = os.path.join(departDir, task, 'mxf')
    filename = os.path.join(renderPath, coder.F.NUKE.Encode(**argv))

    w = nuke.createNode('Write', inpanel=True)
    w['file'].setValue(filename)
    w['file_type'].setValue('mxf')
    w['mxf_op_pattern_knob'].setValue('OP-Atom')

    configData = comm.getDxConfig()
    if configData:
        if 'ACES' in configData['colorSpace']:
            w['colorspace'].setValue('Output - Rec.709')
    else:
        w['colorspace'].setValue('rec709')

    postPython = """
sn = nuke.thisNode()
file = sn['file'].getEvaluatedValue()
wfile = file.replace('.mxf', '_v1.mxf')
os.rename(wfile, file)
"""
    w['afterRender'].setValue(postPython)

# def makeMovWrite(fullPath, isStereo, scsteps):
#     parent = nuke.selectedNode()
#
#     # COMP TEAM PATH
#     startPath = fullPath.split('/script/')[0]
#     jpgDirPath = '_'.join(scsteps)
#     version = scsteps[3]
#     renderPath = "/render/jpg/%s" % jpgDirPath
#     iscomp = True
#
#     if startPath.split('/')[2] == 'lcl':
#         dot1 = nuke.createNode('Dot')
#         dot2 = nuke.createNode('Dot')
#         dot1.setInput(0, parent)
#         dot2.setInput(0, parent)
#         cm = nuke.createNode('CopyMetaData')
#
#         cm.setInput(0, dot1)
#         cm.setInput(1, dot2)
#         ############################################################################################################
#         w = nuke.createNode('Write', inpanel=True)
#         ############################################################################################################
#
#         del scsteps[2]
#         startPath = fullPath.split('/script/')[0]
#         version = scsteps[2]
#         #vnum = version[1:4]
#         jpgDirPath = '_'.join(scsteps[0:2]) + '_' + version
#         renderPath = "/render/jpg"
#         filename = '_'.join(scsteps[0:2]) + '_' + version + '.mov'
#         w.knob('file').setValue(startPath + renderPath + '/' + filename)
#         w['colorspace'].setValue('AlexaV3LogC')
#         w['mov64_fps'].setValue(23.976)

def compRetimeWrite():
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
    elif fullPath.startswith('/mach/'):
        fullPath = fullPath.replace('/mach', '')

    plateImagePath = getTop(nuke.selectedNode())['file'].value()
    plateColorSpace = getTop(nuke.selectedNode())['colorspace'].value()

    coder = rb.Coder()
    argv = coder.D.PLATES.IMAGES.Decode(os.path.dirname(plateImagePath))
    argv2= coder.F.IMAGES.Decode(os.path.basename(plateImagePath))
    argv.update(argv2)
    if 'org' in argv.desc:
        argv.desc = argv.desc.replace('_org', '')
    argv.desc = argv.desc + '_retime'
    # print(argv)
    retimePath = coder.D.PLATES.IMAGES.Encode(**argv)
    fileName = coder.F.IMAGES.BASE.Encode(**argv)

    outputPath = retimePath + '/' + fileName

    w = nuke.createNode('Write')
    w.knob('file').setValue(outputPath)
    w.knob('colorspace').setValue(plateColorSpace)

    # read show _config
    configData = comm.getDxConfig()
    if configData:
        if configData['delivery'].get('format')=='exr':
            w.knob('metadata').setValue("all metadata")
            w['autocrop'].setValue(False)
            # added exr options (from _config)
            for key, value in configData['delivery']['options'].items():
                w.knob(str(key)).setValue(str(value))
        else :
            pass

    if w.knob('file_type').value() == 'dpx' :
        w.knob('datatype').setValue("10 bit")
        w['channels'].setValue('rgb')

    w.setInput(0, w.input(0))
    w.setXYpos(w.xpos() - 0, w.ypos() + 40)

def compWrite(extend):
    jpgColorSpace = None
    if '/' in extend:
        jpgColorSpace = extend.split('/')[1]
        extend = extend.split('/')[0]

    fullPath = nuke.value('root.name')
    print('full : ', fullPath)
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
    elif fullPath.startswith('/mach/'):
        fullPath = fullPath.replace('/mach', '')

    filename = os.path.basename(fullPath)
    coder = rb.Coder()
    argv = coder.D.NUKE.WORKS.Decode(os.path.dirname(fullPath))
    argv2 = coder.F.NUKE.Decode(os.path.basename(fullPath))
    argv.update(argv2)
    print(argv)

    # if '\n' in nuke.root().knob('views').toScript():
    #     isStereo = True
    # else:
    #     isStereo = False

    # add EP number (mov show only )
    if 'mov' in argv.show:
        try:
            import init
            # argv.ep = init.getEP(argv.seq)
            argv.ep = init.getEditOrder('_'.join([argv.seq, argv.shot]))
        except:
            pass

    shotName = '_'.join([argv.seq, argv.shot])
    department = 'comp'
    if extend.startswith('jpg'):
        if argv.departs:
            if 'PFX' in argv.departs:
                department = 'fx'
            elif 'LNR' in argv.departs:
                department = 'lighting'
        elif 'fx' in argv.task:
            department = 'fx'
        # -- lighting --
        # elif 'lighting' in argv.task:
        elif 'lgt' in argv.task:
            department = 'lighting'
        makeJpegWrite(argv.show, argv.seq, shotName, argv.task, department, argv, jpgColorSpace, fullPath)

    elif extend == 'exr':
        makeExrWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    elif extend == 'tiff':
        makeTiffWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    elif extend == 'tiff_mask':
        makeMaskTiffWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    elif extend == 'dpx':
        makeDpxWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    elif extend == 'precomp':
        makePrecompWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    elif extend == 'png':
        makePngWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    elif extend == 'mxf':
        makeMxfWrite(argv.show, argv.seq, shotName, argv.task, department, argv)

    # elif extend == 'mov':
    #     # makeMovWrite(fullPath, isStereo, scsteps)
    #     makeMovWrite(fullPath, scsteps)
