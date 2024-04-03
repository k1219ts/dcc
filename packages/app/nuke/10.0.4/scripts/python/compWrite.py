import os
import string
import re
#import time
import nuke
import nukescripts
import precomp
import stamp
import json


#fullPath : /show/prat/shot/BRDpos/BRDpos_0020/comp/comp/script/BRDpos_0020_comp_v001.nk
#steps :  ['', 'show', 'prat', 'shot', 'BRDpos', 'BRDpos_0020', 'comp', 'comp', 'script', 'BRDpos_0020_comp_v001.nk']
#filename :  BRDpos_0020_comp_v001.nk
#scriptname :  ('BRDpos_0020_comp_v001', '.nk')
#scsteps :  ['BRDpos', '0020', 'comp', 'v001']
#project :  prat
#shotname:  BRDpos_0020
#jobname:  comp
#/show/prat/shot/BRDpos/BRDpos_0020/comp

#/render/comp/exr/BRDpos_0020_comp_v001/BRDpos_0020_comp_%V_v001.%04d.exr

#steps :  ['', 'show', 'prat', 'shot', 'BRDpos', 'BRDpos_0020', 'comp', 'comp', 'script', 'BRDpos_0020_comp_v001.nk']
#filename :  BRDpos_0020_comp_v001.nk
#scriptname :  ('BRDpos_0020_comp_v001', '.nk')
#scsteps :  ['BRDpos', '0020', 'comp', 'v001']
#project :  prat
#shotname:  BRDpos_0020
#jobname:  comp

def getTop(node):
    if node.input(0):
        node = getTop(node.input(0))
    else:
        print "no input"
    return node

# WRITE NODE FOR LOG DI VERSION 1

#------------------------------------------------------------------------------
def logRawPlate():
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
        
    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')    
        
    steps = fullPath.split( os.path.sep )    
    startPath = fullPath.split('/script/')[0]
    exrDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/exr/%s" % exrDirPath
    prj = startPath.split('/')[2]
        
    w= nuke.createNode('Write', inpanel=True)
    filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.exr'
            
    #------------------------------------------------------------------------------ 
    w.knob('colorspace').setValue("linear")
    w.knob('file_type').setValue("exr")   
    w.knob('datatype').setValue("16 bit half")   
    w.knob('compression').setValue("none")
    w.knob('metadata').setValue("all metadata")
    w['channels'].setValue('alpha')
    w['autocrop'].setValue(True)
    w.knob('file').setValue(startPath + renderPath + '/' + filename)

def logFGAlpha():
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
        
    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')    
        
    steps = fullPath.split( os.path.sep )    
    startPath = fullPath.split('/script/')[0]
    exrDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/exr/%s" % exrDirPath
    prj = startPath.split('/')[2]
        
    w= nuke.createNode('Write', inpanel=True)
    filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.exr'
            
    #------------------------------------------------------------------------------ 
    w.knob('colorspace').setValue("linear")
    w.knob('file_type').setValue("exr")   
    w.knob('datatype').setValue("16 bit half")   
    w.knob('compression').setValue("none")
    w.knob('metadata').setValue("all metadata")
    w['channels'].setValue('alpha')
    w['autocrop'].setValue(True)
    w.knob('file').setValue(startPath + renderPath + '/' + filename)   

def logLatest():
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
        
    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')    
        
    steps = fullPath.split( os.path.sep )    
    startPath = fullPath.split('/script/')[0]
    exrDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/exr/%s" % exrDirPath
    prj = startPath.split('/')[2]
        
    w= nuke.createNode('Write', inpanel=True)
    filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.exr'
            
    #------------------------------------------------------------------------------ 
    w.knob('colorspace').setValue("linear")
    w.knob('file_type').setValue("exr")   
    w.knob('datatype').setValue("16 bit half")   
    w.knob('compression').setValue("none")
    w.knob('metadata').setValue("all metadata")
    w['channels'].setValue('rgba')
    w['autocrop'].setValue(True)
    w.knob('file').setValue(startPath + renderPath + '/' + filename)    
 
#------------------------------------------------------------------------------ 
def makeJpegWrite(fullPath, isStereo, scsteps, colorspace):
    paernt = nuke.selectedNode()
    
    iscomp = False
    
    if fullPath.split('/')[6] == 'comp':
        startPath = fullPath.split('/script/')[0]
        jpgDirPath = '_'.join(scsteps)
        version = scsteps[3]
        renderPath = "/render/jpg/%s" % jpgDirPath
        iscomp = True
        
    else:
        startPath = '/'.join(fullPath.split('/')[:-2])
        jpgDirPath = '_'.join(scsteps)
        vn = os.path.basename(fullPath).find('_v')
        dotn = os.path.basename(fullPath).find('.nk')
        version = os.path.basename(fullPath)[vn+1:dotn]
        renderPath = "/jpg/%s" % jpgDirPath

        # if '_w' in os.path.basename(fullPath):
        #     wn = os.path.basename(fullPath).find('_w')
        #     wversion = re.search('w[0-9]+', os.path.basename(fullPath)).group(0)
        #
        #     if wversion in scsteps:
        #         scsteps.remove(wversion)
        #     elif wversion + '.nk' in scsteps:
        #         scsteps.remove(wversion + '.nk')
        #         scsteps.append('.nk')
    
    if startPath.split('/')[2] == 'ssss':
        cs = nuke.createNode('SSSS_viewer_lut')

    elif startPath.split('/')[2] == 'njjl':
        cs = nuke.createNode('NJJL_viewer_lut')

    elif startPath.split('/')[2] == 'mkk3':
        cmValue = nuke.root().knob('colorManagement').value()
        if cmValue == 'Nuke':
            cs = nuke.createNode('mkk3_viewer_lut' , inpanel=False)
        elif cmValue == 'OCIO':
            pass

    elif startPath.split('/')[2] == 'gcd1' and startPath.split('/')[4] == 'ZMA' or startPath.split('/')[4] == 'ZMB':
        cs = nuke.createNode('gcd1_ZMX_lut', inpanel=False)

    elif startPath.split('/')[2] == 'gcd1':
        cs = nuke.createNode('gcd1_viewer_lut', inpanel=False)

    elif startPath.split('/')[2] == 'log':
        ocs = nuke.createNode('OCIOColorSpace')
        ocs['in_colorspace'].setValue('ACES/ACES - ACEScg')
        ocs['out_colorspace'].setValue('Output/Output - Rec.709')

    elif startPath.split('/')[2] == 'god' or startPath.split('/')[2] == 'god2':
        cs = nuke.createNode('god_viewer_lut', inpanel=False)

    elif startPath.split('/')[2] == 'pmc':
        cs = nuke.createNode('pmc_viewer_lut', inpanel=False)

    elif startPath.split('/')[2] == 'trl':
        cs = nuke.createNode('trl_viewer_lut', inpanel=False)

    elif startPath.split('/')[2] == 'ssr':
        cs = nuke.createNode('ssr_viewer_lut', inpanel=False)

    elif startPath.split('/')[2] == 'twe':
        shotName = '_'.join(scsteps[:2])
        jsonData = json.loads(open('/show/twe/_config/nuke/balance_lut.json', 'r').read())
        if jsonData.has_key(shotName):
            cubName = json.loads(open('/show/twe/_config/nuke/balance_lut.json', 'r').read())[shotName]
            lutBasePath = '/show/twe/screening/LUT/TWE_V0825_dexter_TAIKONG_20180827/_Balance_'

            cubFile = os.path.join(lutBasePath, cubName)

            vfNode = nuke.createNode('Vectorfield')
            vfNode['vfield_file'].setValue(cubFile)
            vfNode['colorspaceIn'].setValue('AlexaV3LogC')
            vfNode['colorspaceOut'].setValue('linear')
            vfNode.setName('BALANCE_LUT')
            vfNode['gpuExtrapolate'].setValue(False)

        cs = nuke.createNode('ssr_viewer_lut', inpanel=False)
        nuke.createNode('stamp_twe')

    elif startPath.split('/')[2] == 'btl':
        vfNode = nuke.createNode('Vectorfield')
        vfNode['vfield_file'].setValue("/backstage/apps/Nuke/Globals/Lookup/AlexaV3_EI0800_LogC2Video_Rec709_EE_nuke3d.cube")
        vfNode['colorspaceIn'].setValue('AlexaV3LogC')
        vfNode['colorspaceOut'].setValue('linear')

    elif startPath.split('/')[2] == 'mrz':
        slate = nuke.createNode('slate_idea')
        vfNode = nuke.createNode('Vectorfield')
        vfNode['vfield_file'].setValue("/backstage/apps/Nuke/Globals/Lookup/AlexaV3_K1S1_LogC2Video_Rec709_EE_nuke3d.cube")
        vfNode['colorspaceIn'].setValue('AlexaV3LogC')
        vfNode['colorspaceOut'].setValue('linear')

    elif startPath.split('/')[2] == 'lcl':
        vfNode = nuke.createNode('Vectorfield')
        vfNode['vfield_file'].setValue("/backstage/apps/Nuke/Globals/Lookup/AlexaV3_EI0800_LogC2Video_Rec709_EE_nuke3d.cube")
        vfNode['colorspaceIn'].setValue('AlexaV3LogC')
        vfNode['colorspaceOut'].setValue('linear')

    elif startPath.split('/')[2] == 'srh':
        vfNode = nuke.createNode('Vectorfield')
        vfNode['vfield_file'].setValue("/backstage/apps/Nuke/Globals/Lookup/AlexaV3_EI0800_LogC2Video_Rec709_EE_nuke3d.cube")
        vfNode['colorspaceIn'].setValue('AlexaV3LogC')
        vfNode['colorspaceOut'].setValue('linear')
#subin##
    elif startPath.split('/')[2] == 'mgd':
        pass
        #vfNode = nuke.createNode('Vectorfield')
        #vfNode['vfield_file'].setValue("/stuff/mgd/stuff/LUT/20200311_VFX_RWG_Log3G10_to_REC709.cube")
        #vfNode['colorspaceIn'].setValue('REDLog')
        #vfNode['colorspaceOut'].setValue('linear')


    w= nuke.createNode('Write', inpanel=True)
    
    if isStereo:
        filename = '_'.join(scsteps[0:-1]) + '_%V_' + version + '.%04d.jpg'
             
        w.setXYpos(w.xpos() + 100, w.ypos()+100)
        
        w.knob('views').setValue('right')            
        w.knob('file_type').setValue("jpeg")
        w.knob('_jpeg_quality').setValue(1)
        w.knob('_jpeg_sub_sampling').setValue("4:4:4")
        w.knob('tile_color').setValue(16711935) # set green tile

        wLeft= nuke.createNode('Write', inpanel=True)
        wLeft.setInput(0, w.input(0))       
        wLeft.setXYpos(wLeft.xpos() - 100, wLeft.ypos() + 100)

        wLeft.knob('views').setValue('left')
        wLeft.knob('file_type').setValue("jpeg")
        wLeft.knob('_jpeg_quality').setValue(1)
        wLeft.knob('_jpeg_sub_sampling').setValue("4:4:4")
        wLeft.knob('tile_color').setValue(4278190335) # set red tile
    
        if colorspace == 'sRGB':
            cmValue = nuke.root().knob('colorManagement').value()
            if iscomp:
                renderPath += '_linear'
            
            ##OCIO colorspace setting
            if cmValue == 'OCIO':
                w.knob('colorspace').setValue('Output - Rec.2020')
                wLeft.knob('colorspace').setValue('Output - Rec.2020')
            else :
                w.knob('colorspace').setValue('linear')
                wLeft.knob('colorspace').setValue('linear')
        
        elif colorspace == 'Cineon':
            if iscomp:
                renderPath += '_log'
            w.knob('colorspace').setValue('Cineon')
            wLeft.knob('colorspace').setValue('Cineon')


        if startPath.split('/')[2] == 'xyfy':
            version = 'V2' + str(int(version.split('v')[-1])).zfill(3)
            renderPath = "/render/jpg/%s" % ('_'.join(scsteps[:-1]) + '_' + version).upper()

            lfilename = '_'.join(scsteps[0:-1]).upper() + '_L_' + version + '.%04d.jpg'
            rfilename = '_'.join(scsteps[0:-1]).upper() + '_R_' + version + '.%04d.jpg'


            w.knob('file').setValue(startPath + renderPath + '/' + rfilename)
            wLeft.knob('file').setValue(startPath + renderPath + '/' + lfilename)

        elif startPath.split('/')[2] == 'mkk3':
            #version = 'V2' + str(int(version.split('v')[-1])).zfill(3)
            renderPath = "/render/jpg/%s" % ('_'.join(scsteps[:-1]) + '_' + version)

            lfilename = '_'.join(scsteps[0:2]).upper() + '_' + scsteps[2] + '_L_' + version + '.%04d.jpg'
            rfilename = '_'.join(scsteps[0:2]).upper() + '_' +scsteps[2] + '_R_' + version + '.%04d.jpg'


            w.knob('file').setValue(startPath + renderPath + '/R/' + rfilename)
            wLeft.knob('file').setValue(startPath + renderPath + '/L/' + lfilename)

        else:
            w.knob('file').setValue(startPath + renderPath + '/' + filename)
            wLeft.knob('file').setValue(startPath + renderPath + '/' + filename)
        
    
    else:
        if fullPath.split('/')[6] == 'comp' and startPath.split('/')[2] == 'gcd2' :
            del scsteps[2]
            startPath = fullPath.split('/script/')[0]
            version = scsteps[2]
            vnum = version[1:4]
            jpgDirPath = '_'.join(scsteps[0:2]) + '_' + vnum
            renderPath = "/render/jpg/%s" % jpgDirPath

            filename = '_'.join(scsteps[0:2]) + '_' + vnum + '.%04d.jpg'

        else:
            filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.jpg'

        w.knob('file_type').setValue("jpeg")
        w.knob('_jpeg_quality').setValue(1)
        #w.knob('_jpeg_sub_sampling').setValue("4:4:4")

        if colorspace == 'sRGB':
       	    if startPath.split('/')[2] == 'gcd2' :
                renderPath = renderPath
            else :
                renderPath += '_linear'
            
            cmValue = nuke.root().knob('colorManagement').value()

            ##OCIO colorspace setting
            if startPath.split('/')[2] == 'mkk3':
                w.knob('colorspace').setValue('Output - Rec.2020')
            
            elif startPath.split('/')[2] == 'dok':
                w.knob('colorspace').setValue('Output/Output - Rec.709 (D60 sim.)')

            elif startPath.split('/')[2] == 'ssr':
                w.knob('colorspace').setValue('linear')

            elif startPath.split('/')[2] == 'bds':
                w.knob('colorspace').setValue('Output - Rec.709')

            elif startPath.split('/')[2] == 'cdh':
                w.knob('colorspace').setValue('Output - Rec.709')

            elif startPath.split('/')[2] == 'emd':
                w.knob('colorspace').setValue('Output - Rec.709')

            elif startPath.split('/')[2] == 'mrm':
                w.knob('colorspace').setValue('Output - Rec.709')

            elif startPath.split('/')[2] == 'ban':
                w.knob('colorspace').setValue('Output - Rec.709')

            elif startPath.split('/')[2] == 'srh':
                w.knob('colorspace').setValue('Input - ARRI - Linear - ARRI Wide Gamut')

            elif startPath.split('/')[2] == 'yys':
                w.knob('colorspace').setValue('Output - Rec.709')

            elif startPath.split('/')[2] == 'mgd':
                w.knob('colorspace').setValue('Output - Rec.709')


            else:
                w.knob('colorspace').setValue('linear')

        elif colorspace == 'Cineon':
            renderPath += '_log'
            w.knob('colorspace').setValue('Cineon')
        elif colorspace == 'default(sRGB)':
            w.knob('colorspace').setValue('sRGB')
            w.knob('_jpeg_quality').setValue(1)
            w.knob('_jpeg_sub_sampling').setValue('4:4:4')
            pass

        if startPath.split('/')[2] == 'nmy' :
            w.knob('colorspace').setValue('sRGB')
            w.knob('_jpeg_quality').setValue(1)
            w.knob('_jpeg_sub_sampling').setValue('4:4:4')

        w.knob('file').setValue(startPath + renderPath + '/' + filename)

def makeMeta():
    a= nuke.createNode("ModifyMetaData")
    fulldata = "{set exr/frame_rate 23.976} {set input/frame_rate 23.976}"
    a['metadata'].fromScript(fulldata)



def makeExrWrite(fullPath, isStereo, scsteps):
    startPath = fullPath.split('/script/')[0]
    exrDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/exr/%s" % exrDirPath
    prj = startPath.split('/')[2]
    
    if prj == 'twe':
        stamp = nuke.createNode('stamp_twe')
        stamp['outType'].setValue(1)

    elif prj == 'bds':
        ref = nuke.createNode('Reformat')
        ref['type'].setValue('to format')
        ref['format'].fromScript("2048 858 0 0 2048 858 1 DOK")

    elif prj == 'cdh':
        ref = nuke.createNode('Reformat')
        ref['type'].setValue('to format')
        ref['format'].fromScript("2048 858 0 0 2048 858 1 DOK")

    elif prj == 'emd':
        ref = nuke.createNode('Reformat')
        ref['type'].setValue('to format')
        ref['format'].fromScript("2048 858 0 0 2048 858 1 DOK")

    elif prj == 'srh':
        dd = nuke.createNode('Dot')
        ref = nuke.createNode('Reformat')
        ref['type'].setValue('to format')
        ref['format'].fromScript("1998 1080 0 0 1998 1080 1 PMC_out")
        makeMeta() # modifynodeset

    elif prj == 'yys':
        inputWidth = nuke.selectedNode().width()
        inputHeight = nuke.selectedNode().height()

        if ((inputWidth == 2592) and (inputHeight == 1080)) or ((inputWidth == 3072) and (inputHeight == 1134)):
            re = nuke.createNode('Reformat')
            re['type'].setValue('to box')
            re['resize'].setValue('height')
            re['box_height'].setValue(858)

    w= nuke.createNode('Write', inpanel=True)

    #------------------------------------------------------------------------------

    if isStereo:
        filename = '_'.join(scsteps[0:-1]) + '_%V_' + version + '.%04d.exr'
        w.knob('views').fromScript( '\n'.join(('left', 'right'))) 
    else:
        filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.exr'
            
    #------------------------------------------------------------------------------ 
    cmValue = nuke.root().knob('colorManagement').value()
    if cmValue == 'OCIO':
        w.knob('colorspace').setValue("ACES - ACES2065-1")
    else:
        w.knob('colorspace').setValue("linear")
    w.knob('file_type').setValue("exr")   
    w.knob('datatype').setValue("16 bit half")   
    w.knob('compression').setValue("none")
    w.knob('metadata').setValue("all metadata")

    if prj == 'dok':
        w['channels'].setValue('rgb')
    elif prj == 'prs':
        w['channels'].setValue('rgb')
    elif prj == 'twe':
        w['colorspace'].setValue('AlexaV3LogC')
        w['channels'].setValue('rgba')
        w['autocrop'].setValue(False)
    elif prj == 'bds':
        w['channels'].setValue('rgba')
        w['autocrop'].setValue(True)
    elif prj == 'mrm':
        w['colorspace'].setValue("ACES - ACES2065-1")
    elif prj == 'ban':
        w['channels'].setValue('all')
        w['colorspace'].setValue("ACES - ACES2065-1")
        w['compression'].setValue("PIZ Wavelet (32 scanlines)")
        w['metadata'].setValue("all metadata except input/*")
    elif prj == 'srh':
#        ref = nuke.createNode('Reformat')
#        ref['type'].setValue('to format')
#        ref['format'].fromScript("1998 1080 0 0 1998 1080 1 PMC_out")
        w['colorspace'].setValue('ACES - ACES2065-1')
        w['compression'].setValue('PIZ Wavelet (32 scanlines)')
#		 w['inpanel'].setValue(True)
		 #w= nuke.createNode('Write', inpanel=True,colorspace='ACES - ACES2065-1')
    elif prj == 'yys':
        jfile = '/stuff/yys/stuff/comp/yys_plate_type.json'
        dd = json.load(open(jfile, 'r'))
        print 'YYS WRITE!!!!',scsteps
        shot = scsteps[0] + '_' +  scsteps[1]
        camera_type = dd[shot]
        print shot, camera_type
        #w['channels'].setValue('rgba')
        w['channels'].setValue('rgb')
        #w['compression'].setValue('PIZ Wavelet (32 scanlines)')
        w['compression'].setValue('none')
        w['metadata'].setValue('all metadata except input/*')
        w['colorspace'].setValue('ACES - ACES2065-1')

        """
        if camera_type == 'Sony Venice':
            w['colorspace'].setValue('Input - Sony - S-Log3 - S-Gamut3.Cine')
            pass
        elif camera_type == 'Red Monstro VV':
            w['colorspace'].setValue('Input - RED - REDLog3G10 - REDWideGamutRGB')

            pass
        elif camera_type == 'FULL CG':
            w['colorspace'].setValue('ACES - ACES2065-1')

            pass
        elif camera_type == 'Phantom':
            w['colorspace'].setValue('Output - Rec.709')

        elif camera_type == 'DJI Inspire2':
            w['colorspace'].setValue('Utility - sRGB - Texture')
        """

    else:
        w['channels'].setValue('rgba')
        w['autocrop'].setValue(True)

    w.knob('file').setValue(startPath + renderPath + '/' + filename)
    return w

def makeDpxWrite(fullPath, isStereo, scsteps, isMask=False):
    if isMask:
        scsteps[2] = 'mask'
        startPath = fullPath.split('/script/')[0]
        dpxDirPath = '_'.join(scsteps)
        version = scsteps[3]
        renderPath = "/render/dpx/%s" % dpxDirPath
        if startPath.split('/')[2] == 'tisf':
            # TISF MASK LATER
            return
        
        
        alphaShuffle = nuke.createNode("Shuffle")
        alphaShuffle['red'].setValue('a')
        alphaShuffle['green'].setValue('a')
        alphaShuffle['blue'].setValue('a')
        alphaShuffle['alpha'].setValue('a')

        w= nuke.createNode('Write', inpanel=True)

        print scsteps
        #------------------------------------------------------------------------------
        if isStereo:        
            filename = '_'.join(scsteps[0:-1]) + '_%V_' + version + '.%04d.dpx'
            w.knob('views').fromScript( '\n'.join(('left', 'right'))) 
        else:
            filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'
                       
        #------------------------------------------------------------------------------
        w.knob('colorspace').setValue("Cineon")
        w.knob('channels').setValue('rgba')
        w.knob('file_type').setValue("dpx")   
        w.knob('datatype').setValue("10 bit")
        w.knob('file').setValue(startPath + renderPath + '/' + filename)
        

    else:
        startPath = fullPath.split('/script/')[0]
        dpxDirPath = '_'.join(scsteps)
        version = scsteps[3]
        renderPath = "/render/dpx/%s" % dpxDirPath

        #------------------------------------------------------------------------------
        if isStereo:
            if startPath.split('/')[2] == 'mkk3':
                renderPath = "/render/dpx/%s" % ('_'.join(scsteps[:-1]) + '_' + version)

                lfilename = '_'.join(scsteps[0:2]).upper() + '_' + scsteps[2] + '_L_' + version + '.%04d.dpx'
                rfilename = '_'.join(scsteps[0:2]).upper() + '_' + scsteps[2] + '_R_' + version + '.%04d.dpx'

                format = nuke.createNode('Reformat', inpanel=False)
                format.knob('type').setValue("to format")
                format.knob('format').setValue("2K_DCP")
                format.knob('resize').setValue("none")
                format.knob('black_outside').setValue("1")

                wleft = nuke.createNode('Write', inpanel=True)
                wright = nuke.createNode('Write', inpanel=True)
                wright.setInput(0, wleft.input(0))
                wleft.knob('file').setValue(startPath + renderPath + '/L/' + lfilename)
                wright.knob('file').setValue(startPath + renderPath + '/R/' + rfilename)

                cmValue = nuke.root().knob('colorManagement').value()
                logLutValue = nuke.root().knob('logLut').value()

                ##OCIO setting
                if cmValue == 'OCIO':
                    wleft.knob('colorspace').setValue(logLutValue)
                    wright.knob('colorspace').setValue(logLutValue)
                else:
                    wleft.knob('colorspace').setValue("Cineon")
                    wright.knob('colorspace').setValue("Cineon")
                
            else:
                filename = '_'.join(scsteps[0:-1]) + '_%V_' + version + '.%04d.dpx'
                wleft = nuke.createNode('Write', inpanel=True)
                wright = nuke.createNode('Write', inpanel=True)
                wright.setInput(0, wleft.input(0))
                wleft.knob('file').setValue(startPath + renderPath + '/L/' + filename)
                wright.knob('file').setValue(startPath + renderPath + '/R/' + filename)
                wleft.knob('colorspace').setValue("Cineon")
                wright.knob('colorspace').setValue("Cineon")

            wleft.knob('tile_color').setValue(4278190335)  # set red tile
            wright.knob('tile_color').setValue(16711935)  # set green tile
            wleft.setXYpos(wleft.xpos() - 100, wleft.ypos() + 100)
            wright.setXYpos(wright.xpos() + 100, wright.ypos() + 100)
            wleft.knob('views').setValue('left')
            wleft.knob('file_type').setValue("dpx")
            wleft.knob('datatype').setValue("10 bit")
            wright.knob('views').setValue('right')
            wright.knob('file_type').setValue("dpx")
            wright.knob('datatype').setValue("10 bit")

            seq = dpxDirPath.split('_')[0]
            number = dpxDirPath.split('_')[1]
            vnumber = dpxDirPath.split('_')[3][1:]
            dirName = '%s_%s_%s' % (seq, number, vnumber)
            #startPath + renderPath + '/' + filename)

            return

        else:
            print startPath.split('/')[2]
            if fullPath.split('/')[6] == 'comp' and startPath.split('/')[2] == 'gcd2' :
                startPath = fullPath.split('/script/')[0]
                del scsteps[2]
                version = scsteps[2]
                versionnum = version[1:4]
                dpxDirPath = '_'.join(scsteps[0:2]) + '_' + versionnum
                renderPath = "/render/dpx/%s" % dpxDirPath

                filename = '_'.join(scsteps[0:2]) + '_' + versionnum + '.%04d.dpx'

                format = nuke.nodes.Reformat()
                format.setInput(0, nuke.selectedNode())
                format.setXYpos(format.xpos() - 100, format.ypos() + 50)
                format.knob('type').setValue("to format")
                format.knob('format').setValue("GCD1")
                format.knob('black_outside').setValue("1")

                w = nuke.nodes.Write()
                w.setInput(0, format)
                w.knob('file_type').setValue("dpx")
                w.knob('datatype').setValue("10 bit")

                w.knob('file').setValue(startPath + renderPath + '/' + filename)

            elif startPath.split('/')[2] == 'rom7':
                r = nuke.nodes.Reformat()
                r.setInput(0, nuke.selectedNode())
                r.setXYpos(r.xpos() - 100, r.ypos() + 50)
                r.knob('format').setValue("ROM7")
                w = nuke.nodes.Write()
                w.setInput(0, r)
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'
            
            elif startPath.split('/')[2] == 'god2':
                # addtimeCode for FPS
                if bool(nuke.selectedNode().metadata()) == True:
                    beforeScd = nuke.selectedNode()
                    fhNode = nuke.createNode('FrameHold', inpanel=False)
                    fstFrame = int(nuke.Root()['first_frame'].getValue())
                    fhNode.knob('first_frame').setValue(fstFrame)
                    fstTc = nuke.selectedNode().metadata()['input/timecode']
                    nuke.delete(fhNode)
                    beforeScd.knob('selected').setValue(True)

                    a = nuke.nodes.AddTimeCode()
                    a.setInput(0, nuke.selectedNode())
                    a.setXYpos(a.xpos() - 100, a.ypos() + 50)
                    a.knob('startcode').setValue(fstTc)
                    a.knob('metafps').setValue(False)
                    a.knob('fps').setValue(float(24))
                    a.knob('useFrame').setValue(True)
                    a.knob('frame').setValue(float(1001))
                    a.knob('customPrefix').setValue(True)
                    a.knob('prefix').setValue("dpx/")


                else:
                    a = nuke.nodes.AddTimeCode()
                    a.setInput(0, nuke.selectedNode())
                    a.setXYpos(a.xpos() - 100, a.ypos() + 50)
                    a.knob('startcode').setValue('00:00:00:01')
                    a.knob('metafps').setValue(False)
                    a.knob('fps').setValue(float(24))
                    a.knob('useFrame').setValue(True)
                    a.knob('frame').setValue(float(1001))
                    a.knob('customPrefix').setValue(True)
                    a.knob('prefix').setValue("dpx/")

                r = nuke.nodes.Reformat()
                r.setInput(0, a)
                r.knob('format').setValue("GOD_Out")

                w = nuke.nodes.Write()
                w.setInput(0, r)
                nuke.show(w)
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'

            elif startPath.split('/')[2] == 'mkk3':
                format = nuke.createNode('Reformat', inpanel=False)
                format.knob('type').setValue("to format")
                format.knob('format').setValue("2K_DCP")
                format.knob('resize').setValue("none")
                format.knob('black_outside').setValue("1")

                w = nuke.createNode('Write', inpanel=True)
                w.setInput(0, nuke.selectedNode())
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'

            elif startPath.split('/')[2] == '1987':
                # addtimeCode for FPS
                if bool(nuke.selectedNode().metadata()) == True:
                    beforeScd = nuke.selectedNode()
                    fhNode = nuke.createNode('FrameHold', inpanel=False)
                    fstFrame = int(nuke.Root()['first_frame'].getValue())
                    fhNode.knob('first_frame').setValue(fstFrame)
                    fstTc = nuke.selectedNode().metadata()['input/timecode']
                    nuke.delete(fhNode)
                    beforeScd.knob('selected').setValue(True)

                    a = nuke.nodes.AddTimeCode()
                    a.setInput(0, nuke.selectedNode())
                    a.setXYpos(a.xpos() - 100, a.ypos() + 50)
                    a.knob('startcode').setValue(fstTc)
                    a.knob('metafps').setValue(False)
                    a.knob('fps').setValue(float(23.98))
                    a.knob('useFrame').setValue(True)
                    a.knob('frame').setValue(float(1001))
                    a.knob('customPrefix').setValue(True)
                    a.knob('prefix').setValue("dpx/")

                else:
                    a = nuke.nodes.AddTimeCode()
                    a.setInput(0, nuke.selectedNode())
                    a.setXYpos(a.xpos() - 100, a.ypos() + 50)
                    a.knob('startcode').setValue('00:00:00:01')
                    a.knob('metafps').setValue(False)
                    a.knob('fps').setValue(float(23.98))
                    a.knob('useFrame').setValue(True)
                    a.knob('frame').setValue(float(1001))
                    a.knob('customPrefix').setValue(True)
                    a.knob('prefix').setValue("dpx/")

                r = nuke.nodes.Reformat()
                r.setInput(0, a)
                r.knob('format').setValue('"1987_out"')

                w = nuke.nodes.Write()
                w.setInput(0, r)
                nuke.show(w)
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'

            elif startPath.split('/')[2] == 'nmy':
                r = nuke.nodes.Reformat()
                r.setInput(0, nuke.selectedNode())
                r.setXYpos(r.xpos() - 100, r.ypos() + 50)
                r.knob('format').setValue("NMY_Out")
                w = nuke.nodes.Write()
                w.setInput(0, r)
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'


            elif startPath.split('/')[2] == 'pmc':
                # addtimeCode for FPS
                if bool(nuke.selectedNode().metadata()) == True:
                    beforeScd = nuke.selectedNode()
                    fhNode = nuke.createNode('FrameHold', inpanel=False)
                    fstFrame = int(nuke.Root()['first_frame'].getValue())
                    fhNode.knob('first_frame').setValue(fstFrame)
                    fstTc = nuke.selectedNode().metadata()['input/timecode']
                    nuke.delete(fhNode)
                    beforeScd.knob('selected').setValue(True)
                    """
                    a = nuke.nodes.AddTimeCode()
                    a.setInput(0, nuke.selectedNode())
                    a.setXYpos(a.xpos() - 100, a.ypos() + 50)
                    a.knob('startcode').setValue(fstTc)
                    a.knob('metafps').setValue(False)
                    a.knob('fps').setValue(float(24))
                    a.knob('useFrame').setValue(True)
                    a.knob('frame').setValue(float(1001))
                    a.knob('customPrefix').setValue(True)
                    a.knob('prefix').setValue("dpx/")
                    """


                else:
                    pass
                    """
                    a = nuke.nodes.AddTimeCode()
                    a.setInput(0, nuke.selectedNode())
                    a.setXYpos(a.xpos() - 100, a.ypos() + 50)
                    a.knob('startcode').setValue('00:00:00:01')
                    a.knob('metafps').setValue(False)
                    a.knob('fps').setValue(float(24))
                    a.knob('useFrame').setValue(True)
                    a.knob('frame').setValue(float(1001))
                    a.knob('customPrefix').setValue(True)
                    a.knob('prefix').setValue("dpx/")
                    """
       
                r = nuke.nodes.Reformat()
                r.setInput(0, nuke.selectedNode())
                r.knob('format').setValue("PMC_out")

                w = nuke.nodes.Write()
                w.setInput(0, r)
                nuke.show(w)
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'
                #w['colorspace'].setValue('AlexaV3LogC')

            elif startPath.split('/')[2] == 'saja':
                w = nuke.nodes.Write()
                w.setInput(0, nuke.selectedNode())
                w['colorspace'].setValue('AlexaV3LogC')
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'

            elif startPath.split('/')[2] == 'btl':
                w = nuke.nodes.Write()
                w.setInput(0, nuke.selectedNode())
                w['colorspace'].setValue('AlexaV3LogC')
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'

            elif startPath.split('/')[2] == 'mgd':
                #w = nuke.nodes.Write()
                w = nuke.createNode('Write')
                w.setInput(0, nuke.selectedNode())
                #w['colorspace'].setValue('REDLog')
                print "mgd!!!"
                w['colorspace'].setValue('Input - RED - REDLog3G10 - REDWideGamutRGB')
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'
            else:

                w = nuke.nodes.Write()
                w.setInput(0, nuke.selectedNode())
                nuke.show(w)
                filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.dpx'

        #------------------------------------------------------------------------------

        cmValue = nuke.root().knob('colorManagement').value()
        logLutValue = nuke.root().knob('logLut').value()

        ##OCIO setting
        #if cmValue == 'OCIO':
        #    w.knob('colorspace').setValue(logLutValue)
        #else:
        #    pass
        #    #w.knob('colorspace').setValue("Cineon") 
        
        w.knob('file_type').setValue("dpx")   
        w.knob('datatype').setValue("10 bit")
        w.knob('file').setValue(startPath + renderPath + '/' + filename)

def makePrecompWrite(fullPath, isStereo, scsteps):
#steps :  ['', 'show', 'prat', 'shot', 'BRDpos', 'BRDpos_0020', 'comp', 'comp', 'script', 'BRDpos_0020_comp_v001.nk']
#filename :  BRDpos_0020_comp_v001.nk
#scriptname :  ('BRDpos_0020_comp_v001', '.nk')
#scsteps :  ['BRDpos', '0020', 'comp', 'v001']

    startPath = fullPath.split('/script/')[0]

    version = scsteps[3]
    shotname = '_'.join(scsteps[:2])
    renderPath = "/src/precomp/"
    
    scriptname = shotname + '_precomp_' + version
    if isStereo:
        filename = shotname + '_precomp' + '_%V_' + version + '.%04d.exr'
    else:
        filename = shotname + '_precomp' + '_' + version + '.%04d.exr'
        
    precompUI = precomp.PrecompUI(None, startPath + renderPath,
                                  filename, version)
    if precompUI.exec_():
        w = nuke.nodes.Write()
        if isStereo:
            w.knob('views').fromScript( '\n'.join(('left', 'right')))
        w['channels'].setValue('rgba')
        w['file_type'].setValue('exr')
        w['autocrop'].setValue(True)
        w['file'].setValue(precompUI.result)
        
        w.setInput(0, nuke.selectedNode())

def makePngWrite(fullPath, isStereo, scsteps):

    if isStereo:
        startPath = fullPath.split('/script/')[0]
        pngDirPath = '_'.join(scsteps)
        version = scsteps[3]
        renderPath = "/render/png/%s" % pngDirPath
        filename = '_'.join(scsteps[0:-1]) + '_%V_' + version + '.%04d.png'

        w = nuke.createNode('Write', inpanel=True)
        w.knob('file_type').setValue("png")
        w.knob('channels').setValue('rgba')
        w.knob('colorspace').setValue('sRGB')
        w.knob('file').setValue(startPath + renderPath + '/' + filename)

    else:
        steps = fullPath.split('/')

        startPath = fullPath.split('/script/')[0]
        pngDirPath = '_'.join(scsteps)
        version = scsteps[3]
        renderPath = "/render/png/%s" % pngDirPath
        filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.png'



        if startPath.split('/')[2] == 'gcd1' :
            nuke.createNode('gcd1_viewer_lut', inpanel=False)
            w = nuke.createNode('Write', inpanel=True)
            w.knob('colorspace').setValue("linear")
        else:
            w = nuke.createNode('Write', inpanel=True)
            w.knob('colorspace').setValue('sRGB')

        w.knob('file_type').setValue("png")
        w.knob('channels').setValue('rgba')
        w.knob('file').setValue(startPath + renderPath + '/' + filename)

def makeMaskTiffWrite(fullPath, isStereo, scsteps):
    shotName = scsteps[0] + '_' + scsteps[1]
    startPath = fullPath.split('/script/')[0]
    tiffDirPath = '_'.join(scsteps)
    tiffMaskPath = tiffDirPath + '_mask1'
    version = scsteps[3]
    maskDirPath = shotName + '_comp_' + version
    tiffMaskPath = maskDirPath + '_mask1'
    renderPath = "/render/mask/" + shotName + '/mask1'
    maskFilename = tiffMaskPath + '.%04d' + '.tiff'

    crop = nuke.createNode("Crop", inpanel=False)
    w = nuke.createNode('Write', inpanel=True)
    # ------------------------------------------------------------------------------
    filename = renderPath + '/' + maskFilename
    # ------------------------------------------------------------------------------
    w.knob('file_type').setValue("tiff")
    w.knob('compression').setValue("none")
    w.knob('channels').setValue("rgba")
    w.knob('file').setValue(startPath + filename)

def makeTiffWrite(fullPath, isStereo, scsteps):
    startPath = fullPath.split('/script/')[0]
    tiffDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/tiff/%s" % tiffDirPath
    prj = startPath.split('/')[2]

    crop = nuke.createNode("Crop", inpanel = False)
    w= nuke.createNode('Write', inpanel=True)

    #------------------------------------------------------------------------------
    filename = '_'.join(scsteps[0:-1]) + '_' + version + '.%04d.tiff'

    #------------------------------------------------------------------------------
    w.knob('file_type').setValue("tiff")
    w.knob('compression').setValue("none")
    w.knob('channels').setValue("rgba")
    w.knob('file').setValue(startPath + renderPath + '/' + filename)
##20200608_subin
def yys_matte():
    fullPath = nuke.value('root.name')
    startPath = fullPath.split('/script/')[0]
    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')
    shotName = scsteps[0] + '_' + scsteps[1]
    startPath = fullPath.split('/script/')[0]
    tiffDirPath = '_'.join(scsteps)
    tiffMaskPath = tiffDirPath + '_matte'
    version = scsteps[3]
    maskDirPath = shotName + '_' + scsteps[-2] + '_' + version
    tiffMaskPath = maskDirPath + '_matte'
    renderPath = "/render/matte/" + maskDirPath + '_matte'
    maskFilename = tiffMaskPath + '.%04d' + '.tiff'

    dot0 = nuke.createNode('Dot')
    dot0.setXYpos(dot0.xpos() + 200, dot0.ypos() + 200)
    dot1 = nuke.createNode('Dot')
    dot1.setXYpos(dot0.xpos() - 200, dot0.ypos() + 200)
    dot1.setInput(0,dot0)
    dot2 = nuke.createNode('Dot')
    dot2.setXYpos(dot0.xpos(), dot0.ypos() + 200)
    dot2.setInput(0,dot0)
    dot3 = nuke.createNode('Dot')
    dot3.setXYpos(dot0.xpos() + 200, dot0.ypos() + 200)
    dot3.setInput(0,dot0)
    matte = nuke.createNode('yys_matte.gizmo')
    matte.setXYpos(dot0.xpos()-16 , dot0.ypos() + 400)
    matte.setInput(0,dot1)
    matte.setInput(1,dot2)
    matte.setInput(2,dot3)
    allNode = [dot0,dot1,dot2,dot3,matte]
    for i in allNode:
        i['selected'].setValue(True)
    a = nukescripts.autoBackdrop()
    a['bdheight'].setValue(a['bdheight'].value() + 350)
    a['bdwidth'].setValue(a['bdwidth'].value() + 250)
    a.knob('tile_color').setValue(927764223)
    a.setXYpos(a.xpos() - 100, a.ypos() -100)
    nukescripts.clear_selection_recursive()
    matte['selected'].setValue(True)



    w = nuke.createNode('Write', inpanel=True)
    # ------------------------------------------------------------------------------
    filename = renderPath + '/' + maskFilename
    # ------------------------------------------------------------------------------
    w.knob('file_type').setValue("tiff")
    w.knob('compression').setValue("none")
    w.knob('channels').setValue("rgba")
    w.knob('file').setValue(startPath + filename)
    w.knob('colorspace').setValue('Output - Rec.709')


def makeMovWrite(fullPath, isStereo, scsteps):
    parent = nuke.selectedNode()

    # COMP TEAM PATH
    startPath = fullPath.split('/script/')[0]
    jpgDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/jpg/%s" % jpgDirPath
    iscomp = True

    if startPath.split('/')[2] == 'lcl':
        dot1 = nuke.createNode('Dot')
        dot2 = nuke.createNode('Dot')
        dot1.setInput(0, parent)
        dot2.setInput(0, parent)
        cm = nuke.createNode('CopyMetaData')

        cm.setInput(0, dot1)
        cm.setInput(1, dot2)
        ############################################################################################################
        w = nuke.createNode('Write', inpanel=True)
        ############################################################################################################

        del scsteps[2]
        startPath = fullPath.split('/script/')[0]
        version = scsteps[2]
        #vnum = version[1:4]
        jpgDirPath = '_'.join(scsteps[0:2]) + '_' + version
        renderPath = "/render/jpg"
        filename = '_'.join(scsteps[0:2]) + '_' + version + '.mov'
        w.knob('file').setValue(startPath + renderPath + '/' + filename)
        w['colorspace'].setValue('AlexaV3LogC')
        w['mov64_fps'].setValue(23.976)



#set layer name by giuk
def namePanel():
    nPanel = nuke.Panel("set Layer name")
    nPanel.addSingleLineInput("Layer_Name : ", "")
    retVar = nPanel.show()
    nameVar = nPanel.value("Layer_Name : ")
    return (retVar,nameVar)

def TIFF_DOK():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]

        fullPath = nuke.value('root.name')
        startPath = fullPath.split('/script/')[0]
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        del scsteps[2]
        version = scsteps[2]
        dpxDirPath = '_'.join(scsteps[0:2]) + '_' + layerName + '_' + version
        layerPath = dpxDirPath + '_' + layerName + '/'
        renderPath = "/render/mask/%s/" % dpxDirPath
        filename = dpxDirPath + '.%04d.tiff'
        print startPath + renderPath + filename
        s = nuke.nodes.Shuffle()
        s.setInput(0, nuke.selectedNode())
        s.knob('in').setValue("alpha")

        w = nuke.nodes.Write()
        w.setInput(0, s)
        w.knob('file_type').setValue("tiff")
        w.knob('compression').setValue("none")
        w.knob('channels').setValue("rgba")
        w.knob('colorspace').setValue("ACES - ACES2065-1")
        w.knob('file').setValue(startPath + renderPath + filename)


def TIFF_TWE():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]

        fullPath = nuke.value('root.name')
        startPath = fullPath.split('/script/')[0]
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        del scsteps[2]
        version = scsteps[2]
        #versionnum = version[1:4]
        dpxDirPath = '_'.join(scsteps[0:2]) + '_' + layerName + '_' + version + '/'
        print "dpxDirPath", dpxDirPath
        #layerPath = dpxDirPath + '_' + layerName + '/'
        renderPath = "/render/mask/"
        filename = '_'.join(scsteps[0:2]) + '_' + layerName + '_' + version + '.%04d.tiff'
        print startPath + renderPath + dpxDirPath + filename

        stamp = nuke.createNode('twe_alpha_stamp')
        
        #format = nuke.nodes.Reformat()
        #format.setInput(0, nuke.selectedNode())
        #format.setXYpos(format.xpos() - 100, format.ypos() + 50)
        #format.knob('type').setValue("to format")
        #format.knob('format').setValue("GCD1")
        #format.knob('black_outside').setValue("1")

        w = nuke.nodes.Write()
        w.setInput(0, stamp)
        w.knob('file_type').setValue("tiff")
        w.knob('compression').setValue("none")
        w.knob('channels').setValue("rgba")
        w.knob('file').setValue(startPath + renderPath + dpxDirPath + filename)

    else:
        print ('CANCELLED')


# GCD1 project write setting by giuk
def TIFF_GCD1():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]

        fullPath = nuke.value('root.name')
        startPath = fullPath.split('/script/')[0]
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        del scsteps[2]
        version = scsteps[2]
        versionnum = version[1:4]
        dpxDirPath = '_'.join(scsteps[0:2]) + '_' + versionnum
        layerPath = dpxDirPath + '_' + layerName + '/'
        renderPath = "/render/mask/%s/" % dpxDirPath
        filename = '_'.join(scsteps[0:2]) + '_' + versionnum + '_' + layerName + '.%04d.tiff'
        print startPath + renderPath + layerPath + filename

        format = nuke.nodes.Reformat()
        format.setInput(0, nuke.selectedNode())
        format.setXYpos(format.xpos() - 100, format.ypos() + 50)
        format.knob('type').setValue("to format")
        format.knob('format').setValue("GCD1")
        format.knob('black_outside').setValue("1")

        w = nuke.nodes.Write()
        w.setInput(0, format)
        w.knob('file_type').setValue("tiff")
        w.knob('compression').setValue("none")
        w.knob('channels').setValue("rgba")
        w.knob('file').setValue(startPath + renderPath + layerPath + filename)

    else:
        print ('CANCELLED')

# GOD project write setting by giuk
def TIFF_GOD():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]

        fullPath = nuke.value('root.name')
        startPath = fullPath.split('/script/')[0]
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        del scsteps[2]
        shotname = '_'.join(scsteps[0:2])
        version = scsteps[2]
        dpxDirPath = shotname + '_' + layerName + '_' + version
        renderPath = "/render/mask/" + shotname +"/" + "%s/" % dpxDirPath
        filename = dpxDirPath + '.%04d.tiff'
        print startPath + renderPath + filename

        format = nuke.nodes.Reformat()
        format.setInput(0, nuke.selectedNode())
        format.setXYpos(format.xpos() - 100, format.ypos() + 50)
        format.knob('type').setValue("to format")
        format.knob('format').setValue("GOD_Out")
        format.knob('black_outside').setValue("1")

        alphaShuffle = nuke.nodes.Shuffle()
        alphaShuffle.setInput(0, format)
        alphaShuffle['red'].setValue('a')
        alphaShuffle['green'].setValue('a')
        alphaShuffle['blue'].setValue('a')
        alphaShuffle['alpha'].setValue('a')

        w = nuke.nodes.Write()
        w.setInput(0, alphaShuffle)
        w.knob('file_type').setValue("tiff")
        w.knob('compression').setValue("none")
        w.knob('channels').setValue("rgba")
        w.knob('file').setValue(startPath + renderPath + filename)

    else:
        print ('CANCELLED')

def TIFF_MKK3():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]
        fullPath = nuke.value('root.name')
        if fullPath.startswith('/netapp/dexter'):
            fullPath = fullPath.replace('/netapp/dexter', '')
        steps = fullPath.split(os.path.sep)
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        startPath = fullPath.split('/script/')[0]
        print scsteps
        version = scsteps[3]
        lrenderPath = "/render/matte/L/%s/" % ('_'.join(scsteps[0:2]).upper() + '_' + layerName + '_' + version)
        rrenderPath = "/render/matte/R/%s/" % ('_'.join(scsteps[0:2]).upper() + '_' + layerName + '_' + version)
        lfilename = '_'.join(scsteps[0:2]).upper() + '_' + layerName + '_L_' + version + '.%04d.tiff'
        rfilename = '_'.join(scsteps[0:2]).upper() + '_' + layerName + '_R_' + version + '.%04d.tiff'

        format = nuke.createNode('Reformat', inpanel=False)
        format.knob('type').setValue("to format")
        format.knob('format').setValue("2K_DCP")
        format.knob('resize').setValue("none")
        format.knob('black_outside').setValue("1")

        wleft = nuke.createNode('Write', inpanel=True)
        wright = nuke.createNode('Write', inpanel=True)
        wright.setInput(0, wleft.input(0))
        wleft.knob('file').setValue(startPath + lrenderPath + lfilename)
        wright.knob('file').setValue(startPath + rrenderPath + rfilename)
        
        wleft.knob('tile_color').setValue(4278190335)  # set red tile
        wright.knob('tile_color').setValue(16711935)  # set green tile
        wleft.setXYpos(wleft.xpos() - 100, wleft.ypos() + 100)
        wright.setXYpos(wright.xpos() + 100, wright.ypos() + 100)


        wleft.knob('views').setValue('left')
        wleft.knob('file_type').setValue("tiff")
        wleft.knob('compression').setValue("none")
        wleft.knob('channels').setValue("rgba")
        wright.knob('views').setValue('right')
        wright.knob('file_type').setValue("tiff")
        wright.knob('compression').setValue("none")
        wright.knob('channels').setValue("rgba")
    else:
        print ('CANCELLED')



def GCD1_Layer():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]

        fullPath = nuke.value('root.name')
        startPath = fullPath.split('/script/')[0]
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        del scsteps[2]
        version = scsteps[2]
        versionnum = version[1:4]
        dpxDirPath = '_'.join(scsteps[0:2]) + '_' + versionnum
        layerPath = dpxDirPath + '_' + layerName + '/'
        renderPath = "/render/layer/%s/" % dpxDirPath
        filename = '_'.join(scsteps[0:2]) + '_' + versionnum + '_' + layerName + '.%04d.dpx'
        print startPath + renderPath + layerPath + filename

        format = nuke.nodes.Reformat()
        format.setInput(0, nuke.selectedNode())
        format.setXYpos(format.xpos() - 100, format.ypos() + 50)
        format.knob('type').setValue("to format")
        format.knob('format').setValue("GCD1")
        format.knob('black_outside').setValue("1")

        w = nuke.nodes.Write()
        w.setInput(0, format)
        w.knob('file_type').setValue("dpx")
        w.knob('datatype').setValue("10 bit")
        w.knob('channels').setValue("rgba")
        w.knob('file').setValue(startPath + renderPath + layerPath + filename)

    else:
        print ('CANCELLED')

def MKK3_Layer():
    panelValue = namePanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        layerName = panelValue[1]
        fullPath = nuke.value('root.name')
        if fullPath.startswith('/netapp/dexter'):
            fullPath = fullPath.replace('/netapp/dexter', '')
        steps = fullPath.split(os.path.sep)
        filename = os.path.basename(fullPath)
        scriptname = os.path.splitext(filename)
        scsteps = scriptname[0].split('_')
        startPath = fullPath.split('/script/')[0]
        print scsteps
        version = scsteps[3]
        renderPath = "/render/dpx/%s" % ('_'.join(scsteps[0:2]).upper() + '_' + layerName + '_' + version)
        lfilename = '_'.join(scsteps[0:2]).upper() + '_' + layerName + '_L_' + version + '.%04d.dpx'
        rfilename = '_'.join(scsteps[0:2]).upper() + '_' + layerName + '_R_' + version + '.%04d.dpx'
        format = nuke.createNode('Reformat', inpanel=False)
        format.knob('type').setValue("to format")
        format.knob('format').setValue("2K_DCP")
        format.knob('resize').setValue("none")
        format.knob('black_outside').setValue("1")
        wleft = nuke.createNode('Write', inpanel=True)
        wright = nuke.createNode('Write', inpanel=True)
        wright.setInput(0, wleft.input(0))
        wleft.knob('file').setValue(startPath + renderPath + '/L/' + lfilename)
        wright.knob('file').setValue(startPath + renderPath + '/R/' + rfilename)
        cmValue = nuke.root().knob('colorManagement').value()
        logLutValue = nuke.root().knob('logLut').value()
        ##OCIO setting
        if cmValue == 'OCIO':
            wleft.knob('colorspace').setValue(logLutValue)
            wright.knob('colorspace').setValue(logLutValue)
        else:
            wleft.knob('colorspace').setValue("Cineon")
            wright.knob('colorspace').setValue("Cineon")
        wleft.knob('tile_color').setValue(4278190335)  # set red tile
        wright.knob('tile_color').setValue(16711935)  # set green tile
        wleft.setXYpos(wleft.xpos() - 100, wleft.ypos() + 100)
        wright.setXYpos(wright.xpos() + 100, wright.ypos() + 100)
        wleft.knob('views').setValue('left')
        wleft.knob('file_type').setValue("dpx")
        wleft.knob('datatype').setValue("10 bit")
        wleft.knob('channels').setValue("rgba")
        wright.knob('views').setValue('right')
        wright.knob('file_type').setValue("dpx")
        wright.knob('datatype').setValue("10 bit")
        wright.knob('channels').setValue("rgba")
    else:
        print ('CANCELLED')


def TIFF_ASD():
    fullPath = nuke.value('root.name')
    print 'full : ', fullPath
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')

    steps = fullPath.split(os.path.sep)
    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')

    startPath = fullPath.split('/script/')[0]
    tiffDirPath = '_'.join(scsteps)
    version = scsteps[3]
    renderPath = "/render/tiff/%s" % tiffDirPath
    prj = startPath.split('/')[2]

    crop = nuke.createNode("Crop", inpanel = False)
    w= nuke.createNode('Write', inpanel=True)

    #------------------------------------------------------------------------------
    filename = '_'.join(scsteps[0:-1]) + '_' + version + '_mask.%04d.tiff'

    #------------------------------------------------------------------------------
    w.knob('file_type').setValue("tiff")
    w.knob('compression').setValue("none")
    w.knob('channels').setValue("rgba")
    w.knob('file').setValue(startPath + renderPath + '/' + filename)

    '/show/asd01/shot/S39/S39_0740/comp/comp/render/tiff/S39_0740_comp_v010/S39_0740_comp_v010_mask.%04d.tiff '
    '/show/asd04/shot/S41/S41_0050/comp/comp/render/tiff/S41_0050_comp_v001/S41_0050_comp_v001.%04d.tiff'


# SSSS project write setting by giuk
def DPX_SSSS():

    # fullcg EX) EB_0010 / EB_0010_V001_comp.1001.dpx
    # palte EX) EB_0020 / 011_02_04A / A091_C012_02184G / EB_0020_V001_comp.1001.dpx

    # dpxfilename
    nkFilePath = nuke.root().name()

    if nkFilePath.startswith('/netapp/dexter'):
        nkFilePath = nkFilePath.replace('/netapp/dexter', '')

    shotName = nkFilePath.split('/')[5]
    startPath = nkFilePath.split('/script/')[0]
    ScriptName = nkFilePath.split('/')[9]
    versteps = ScriptName.split('.')[0]
    shotVer = versteps.upper().split('_')[3]
    dpxFileName = shotName + '_' + shotVer + '_' + 'comp' + '.' + '%04d' + '.' + 'dpx'
    cgDpxPath = shotName + '_' + shotVer + '_' + 'comp'
    dpxPath = startPath + '/render/dpx/'
    dirName = ScriptName.split('.')[0]

    # json import
    reelData = json.loads(open('/netapp/dexter/show/ssss/stuff/reel/ssss_reel.json', 'r').read())
    shotValue = reelData[shotName]
    reelname = shotValue['reelname']
    filename = shotValue['filename']

    # FullPath_Setting
    fullcgPath = dpxPath + os.path.join(dirName, dpxFileName)
    platePath = dpxPath + os.path.join(dirName, reelname, filename, dpxFileName)

    # writeNode_fuction
    def WriteNodeCreate(fullPath):

	#addtimeCode for FPS

        beforeScd = nuke.selectedNode()
        fhNode = nuke.createNode('FrameHold', inpanel=False)
        fstFrame = int(nuke.Root()['first_frame'].getValue())
        fhNode.knob('first_frame').setValue(fstFrame)
        fstTc = nuke.selectedNode().metadata()['input/timecode']
        nuke.delete(fhNode)
        beforeScd.knob('selected').setValue(True)

        a = nuke.createNode('AddTimeCode')
        a.knob('startcode').setValue(fstTc)
        a.knob('metafps').setValue(False)
        a.knob('fps').setValue(float(23.98))
        a.knob('useFrame').setValue(True)
        a.knob('frame').setValue(float(1001))
        a.knob('customPrefix').setValue(True)
        a.knob('prefix').setValue("dpx/")
        # ----------------------------------------------------------
        
	w = nuke.createNode('Write', inpanel=True)

        # ----------------------------------------------------------
        w.knob('colorspace').setValue("linear")
        w.knob('file_type').setValue("dpx")
        w.knob('datatype').setValue("10 bit")
        w.knob('file').setValue(fullPath)
        return w

    if False in shotValue.values():
        WriteNodeCreate(platePath)
    else:
        WriteNodeCreate(fullcgPath)

def TIFF_SSSS():
    nkFilePath = nuke.root().name()

    if nkFilePath.startswith('/netapp/dexter'):
        nkFilePath = nkFilePath.replace('/netapp/dexter', '')

    shotName = nkFilePath.split('/')[5]

    startPath = nkFilePath.split('/script/')[0]
    ScriptName = nkFilePath.split('/')[9]
    versteps = ScriptName.split('.')[0]
    shotVer = versteps.lower().split('_')[3]
    tiffFileName = shotName + '_' + 'comp' + '_' + shotVer + '_' + 'mask1' + '.' + '%04d' + '.' + 'tiff'
    cgDpxPath = shotName + '_' + shotVer + '_' + 'comp'
    tiffRootPath = startPath + '/render/tiff/'
    dirName = ScriptName.split('.')[0]

    # json import
    reelData = json.loads(open('/netapp/dexter/show/ssss/stuff/reel/ssss_reel.json', 'r').read())
    shotValue = reelData[shotName]
    reelname = shotValue['reelname']

    # FullPath_Setting
    tiffPath = tiffRootPath + shotName + '/' + reelname + '/' + 'MASK' + '/' + 'MASK1' + '/' + tiffFileName


    # writeNode_fuction
    def tiffWriteCreate(fullPath):
        crop = nuke.createNode("Crop", inpanel=False)
        s = nuke.createNode("Shuffle", inpanel=False)
        w = nuke.createNode('Write', inpanel=True)

        # ----------------------------------------------------------
        s.knob('in').setValue('alpha')
        w.knob('file_type').setValue("tiff")
        w.knob('compression').setValue("none")
        w.knob('file').setValue(fullPath)
        w.knob('channels').setValue("rgba")
        return w

    if False in shotValue.values():
        tiffWriteCreate(tiffPath)
    else:
        tiffWriteCreate(tiffPath)

def EXR_SSSS():
    nkFilePath = nuke.root().name()

    if nkFilePath.startswith('/netapp/dexter'):
        nkFilePath = nkFilePath.replace('/netapp/dexter', '')

    shotName = nkFilePath.split('/')[5]

    startPath = nkFilePath.split('/script/')[0]
    ScriptName = nkFilePath.split('/')[9]
    versteps = ScriptName.split('.')[0]
    shotVer = versteps.lower().split('_')[3]
    exrFileName = shotName + '_' + 'comp' + '_' + shotVer + '_' + 'fg' + '.' + '%04d' + '.' + 'exr'
    cgDpxPath = shotName + '_' + shotVer + '_' + 'comp'
    exrRootPath = startPath + '/render/exr/'
    dirName = ScriptName.split('.')[0]

    # json import
    reelData = json.loads(open('/netapp/dexter/show/ssss/stuff/reel/ssss_reel.json', 'r').read())
    shotValue = reelData[shotName]
    reelname = shotValue['reelname']
    # filename = shotValue['filename']

    # FullPath_Setting
    # fullcgPath = tiffPath + shotName + '/' + 'MASK' + '/' + tiffFileName
    exrPath = exrRootPath + shotName + '/' + reelname + '/' + 'LAYERS' + '/' + 'FG' + '/' + exrFileName

    # writeNode_fuction
    def exrCreate(fullPath):
        w = nuke.createNode('Write', inpanel=True)
        # ------------------------------------------------------------------------------
        w.knob('colorspace').setValue("linear")
        w.knob('file_type').setValue("exr")
        w.knob('datatype').setValue("16 bit half")
        w.knob('compression').setValue("none")
        w.knob('metadata').setValue("all metadata")
        w['channels'].setValue('rgba')
        w['autocrop'].setValue(True)
        w.knob('file').setValue(fullPath)
        return w

    if False in shotValue.values():
        exrCreate(exrPath)
    else:
        exrCreate(exrPath)


#compRetimeWrite by BYUNGCHAN
def compRetimeWrite():
    fullPath_org = nuke.root().name()

    if fullPath_org.startswith('/netapp/'):
        fullPath_list = fullPath_org.split('/')
        del fullPath_list[0:3]
        fullPath = '/' + '/'.join(fullPath_list)
        # print fullPath
    else:
        fullPath = fullPath_org
        # print fullPath

    vsteps = fullPath.split('retime_')[1]
    # print vsteps

    version = vsteps[0:2] + vsteps[3]


    GTP_org = getTop(nuke.selectedNode())['file'].value()
    GTP_colorSpace = getTop(nuke.selectedNode())['colorspace'].value()

    if GTP_org.startswith('/netapp/'):
        GTP_list = GTP_org.split('/')
        del GTP_list[0:3]
        GTP = '/' + '/'.join(GTP_list)
        print GTP
    else:
        GTP = GTP_org
        print GTP

    csteps = GTP.split('.')

    contain = csteps[-2] + '.' + csteps[-1]

    ppsteps = '/'.join(GTP.split('/')[0:7])

    ppsteps2 = GTP.split('/')[7].split('_')[0]

    startPath = ppsteps + '/' + ppsteps2

    retimePPath = startPath + '_retime' + '/' + version

    fsteps = retimePPath.split('/')

    if '\n' in nuke.root().knob('views').toScript():
        isStereo = True
    else:
        isStereo = False

    if isStereo:
        fileName = fsteps[5] + '_' + fsteps[7] + '_%V_' + fsteps[8] + '.' + contain
    else:
        fileName = fsteps[5] + '_' + fsteps[7] + '_' + fsteps[8] + '.' + contain

    retimePath = retimePPath + '/' + fileName

    w = nuke.createNode('Write')

    w.knob('file').setValue(retimePath)

    w.knob('file').setValue(retimePath)

    if startPath.split('/')[2] == 'mkk3' or startPath.split('/')[2] == 'gcd1' :
        w.knob('colorspace').setValue("Cineon")

    else :
        w.knob('colorspace').setValue(GTP_colorSpace)

    
    if w.knob('file_type').value() == 'exr' :
        w.knob('datatype').setValue("16 bit half")
        w.knob('compression').setValue("none")
        w.knob('metadata').setValue("all metadata")
        w['channels'].setValue('rgb')
        w['autocrop'].setValue(True)

    elif w.knob('file_type').value() == 'dpx' :
        w.knob('datatype').setValue("10 bit")
        w['channels'].setValue('rgb')

    w.setInput(0, w.input(0))
    w.setXYpos(w.xpos() - 0, w.ypos() + 40)




def compWrite(extend):
    'jpg/default(sRGB)'
    #'jpg/linear' jpg/sRGB'
    #'exr' 'mov'
    #precomp
    
    #===============================================================================
    # Prepare components for write
    #===============================================================================
    jpgColorSpace = None
    if '/' in extend:
        jpgColorSpace = extend.split('/')[1]
        extend = extend.split('/')[0]

    fullPath = nuke.value('root.name')
    print 'full : ', fullPath
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
    elif fullPath.startswith('/mach/'):
        fullPath = fullPath.replace('/mach', '')
        
    steps = fullPath.split( os.path.sep )

    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')
    
    project = string.join(steps[0:4], '/')
    
    sequence = steps[4]
    shotname = steps[5]
    project = steps[2] # Will be used futher
    jobname = scsteps[2]
    
    if '\n' in nuke.root().knob('views').toScript():
        isStereo = True
    else:
        isStereo = False
            
#    print 'fullPath : ', fullPath
#    print 'steps : ', steps
#    print 'filename : ', filename
#    print 'scriptname : ', scriptname
#    print 'scsteps : ', scsteps
#    print 'project : ', project
#    print 'shotname: ', shotname
#    print 'jobname: ', jobname
    
    if extend.startswith('jpg'):     
        makeJpegWrite(fullPath, isStereo, scsteps, jpgColorSpace)
    
    elif extend == 'exr':
        makeExrWrite(fullPath, isStereo, scsteps)

    elif extend == 'tiff':
        makeTiffWrite(fullPath, isStereo, scsteps)

    elif extend == 'tiff_mask':
        makeMaskTiffWrite(fullPath, isStereo, scsteps)
                
    elif extend == 'dpx':
        makeDpxWrite(fullPath, isStereo, scsteps)

    elif extend == 'precomp':
         makePrecompWrite(fullPath, isStereo, scsteps)
         
    elif extend == 'png':
         makePngWrite(fullPath, isStereo, scsteps)

    elif extend == 'mov':
         makeMovWrite(fullPath, isStereo, scsteps)
