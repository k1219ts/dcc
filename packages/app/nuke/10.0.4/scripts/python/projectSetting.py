import nuke, json
import sys, os

testFps = 23.976
testFps2 = 23.98


def settingYYS():   
     ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')
    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES - ACES2065-1')



#--------------jhjung
def settingMGD():
    ## ACES - OCIO project Setting
    #nuke.root().knob('colorManagement').setValue('Nuke')
    #nuke.root().knob('OCIO_config').setValue('nuke-default')

    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
#            i['viewerProcess'].setValue('MGD')
            i['viewerProcess'].setValue('Rec.709 (ACES)')

    else:
#        nuke.createNode('Viewer').knob('viewerProcess').setValue('MGD')
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')


    nuke.knobDefault("Viewer.viewerProcess", 'MGD')

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('Input - RED - REDLog3G10 - REDWideGamutRGB')
    nuke.root()['format'].setValue('2K_DCP')
    nuke.knobDefault("Root.format", "2K_DCP")

#--------------------------------------------

#0316 subin
testFps = 23.976
def settingSRH():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('aces_1.0.1')
    #nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')
    nuke.root().knob('workingSpaceLUT').setValue('Input - ARRI - Linear - ARRI Wide Gamut')

    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('AlexaV3Rec709')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('AlexaV3Rec709')

    nuke.knobDefault("Viewer.viewerProcess", 'AlexaV3Rec709')
    nuke.root()['format'].setValue("PMC_out")
    nuke.root()['fps'].setValue(testFps)


def settingIMT():
    fullPath = nuke.root().name()
    if fullPath.startswith('/netapp/show/'):
        fullPath = fullPath.replace('/netapp/show', '')
    print "fullpath : ", fullPath

    nuke.knobDefault("Root.format", "IMT")
    nuke.root()['format'].setValue('IMT')

    try:
        shotLutDic  = json.load(open('/show/imt/stuff/post/_LUT/IMT_platelist.json', 'r'))
        lutBasePath = '/show/imt/stuff/post/_LUT/ccc'
        # CHECK LUT FROM SEQ
        seq = fullPath.split('/')[4]
        shot = fullPath.split('/')[5]

        if shotLutDic.has_key(shot):
            plates = shotLutDic[shot].keys()
            for plate in reversed(plates):

                plateName = shotLutDic[shot][plate]

                nuke.ViewerProcess.register("imt_%s_%s" % (shot, plate),
                                            nuke.createNode,
                                            ("imt_cdl_lut", "shotName %s plate %s" % (shot, plate))
                                            )
            nuke.knobDefault("Viewer.viewerProcess", "imt_%s_%s" % (shot, plate))

    except:
        pass
    nuke.addFormat('2048 1152 IMT')

def settingBDS():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES - ACES2065-1')
    nuke.root()['format'].setValue('BDS')
    nuke.knobDefault("Root.format", "BDS")


#subin 0720 temporary
def settingCDH():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES - ACES2065-1')
    nuke.root()['format'].setValue('BDS')
    nuke.knobDefault("Root.format", "BDS")
    nuke.root()['fps'].setValue(testFps2)
#2020.08.05 subin temporary
def settingEMD():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES - ACES2065-1')
    nuke.root()['format'].setValue('DOK')
    nuke.knobDefault("Root.format", "DOK")
    nuke.root()['fps'].setValue(testFps2)


def settingVGD():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')

def settingMRZ():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('MRZ')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('MRZ')

    nuke.knobDefault("Viewer.viewerProcess", "MRZ")
    #nuke.root()['format'].setValue('HOLO')
    #nuke.knobDefault("Root.format", "HOLO")
    nuke.root()['logLut'].setValue('AlexaV3LogC')



def settingHOL():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('HOLO')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('HOLO')

    nuke.knobDefault("Viewer.viewerProcess", "HOLO")
    nuke.root()['format'].setValue('HOLO')
    nuke.knobDefault("Root.format", "HOLO")


def settingBTL():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('AlexaV3Rec709')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('AlexaV3Rec709')

    nuke.knobDefault("Viewer.viewerProcess", "AlexaV3Rec709")
    nuke.root()['format'].setValue('GOD')
    nuke.knobDefault("Root.format", "GOD")

def settingSAJA():
    fullPath = nuke.root().name()
    if fullPath.startswith('/netapp/show/'):
        fullPath = fullPath.replace('/netapp/show', '')
    print "fullpath : ", fullPath

    try:
        shotLutDic  = json.load(open('/stuff/saja/stuff/lut/20190123_LUT.json', 'r'))
        lutBasePath = '/stuff/saja/stuff/lut/ALL'
        # CHECK LUT FROM SEQ
        seq = fullPath.split('/')[4]
        shot = fullPath.split('/')[5]

        if shot in ['JMS_0010']:
            nuke.ViewerProcess.register("SAJA_JMS_0010",
                                        nuke.createNode,
                                        ("JMS_0010_lut", "")
                                        )
            nuke.knobDefault("Viewer.viewerProcess", "SAJA_JMS_0010")
            for i in nuke.allNodes('Viewer'):
                px = i.xpos()
                py = i.ypos()
                input_backup = []
                for j in range(i.inputs()):
                    input_backup.append(i.input(j))

                nuke.delete(i)
                v = nuke.nodes.Viewer()
                v['viewerProcess'].setValue("SAJA_JMS_0010")
                v.setXYpos(px, py)
                for j in range(len(input_backup)):
                    v.setInput(j, input_backup[j])

        elif shot == 'YHR_0010':
            lutFilePath = os.path.join(lutBasePath, shotLutDic[shot]) + '.cube'
            print "lutFilePath", lutFilePath
            nuke.ViewerProcess.register("SAJA_%s" % shot,
                                        nuke.createNode,
                                        ("Vectorfield",
                                         "vfield_file %s colorspaceIn linear colorspaceOut linear" % lutFilePath)
                                        )

            allViewer = nuke.allNodes('Viewer')
            if allViewer:
                for i in allViewer:
                    i['viewerProcess'].setValue("SAJA_%s" % shot)
            else:
                nuke.createNode('Viewer').knob('viewerProcess').setValue("SAJA_%s" % shot)

        elif shot in ['JFT_0010', 'JFT_0020', 'JFT_0040', 'JFT_0060', 'JFT_0070', 'JFT_0080', 'JFT_0100', 'JFT_0120', 'JFT_0130', 'JFT_0140', 'JFT_0160', 'JFT_0180', 'JFT_0190', 'JFT_0200', 'JFT_0210', 'JFT_0220', 'JFT_0230', 'JFT_0240', 'JFT_0250', 'JFT_0270', 'JFT_0290', 'JFT_0310', 'JFT_0320', 'JFT_0330', 'JFT_0340', 'JFT_0350', 'JFT_0370', 'JFT_0380', 'JFT_0390', 'JFT_0400', 'JFT_0410', 'JFT_0420', 'JFT_0440', 'JFT_0450', 'JFT_0460', 'JFT_0470', 'JFT_0480', 'JFT_0490', 'JFT_0500', 'JFT_0510', 'JFT_0520', 'JFT_0540', 'JFT_0550', 'JFT_0560', 'JFT_0570', 'JFT_0580', 'JFT_0590', 'JFT_0630', 'JFT_0640', 'JFT_0650', 'JFT_0660', 'JFT_0670', 'JFT_0680', 'JFT_0690', 'JFT_0700', 'JFT_0720', 'JFT_0740', 'JFT_0750', 'JFT_0760', 'JFT_0770', 'JFT_0780', 'JFT_0800', 'JFT_0820', 'JFT_0830', 'JFT_0840', 'JFT_0860', 'JFT_0870', 'JFT_0880', 'JFT_0890', 'JFT_0900']:
            nuke.ViewerProcess.register("SAJA_%s" % shot,
                                        nuke.createNode,
                                        (shot, ""))
            nuke.knobDefault("Viewer.viewerProcess", "SAJA_%s" % shot)
            for i in nuke.allNodes('Viewer'):
                px = i.xpos()
                py = i.ypos()
                input_backup = []
                for j in range(i.inputs()):
                    input_backup.append(i.input(j))

                nuke.delete(i)
                v = nuke.nodes.Viewer()
                v['viewerProcess'].setValue("SAJA_%s" % shot)
                v.setXYpos(px, py)
                for j in range(len(input_backup)):
                    v.setInput(j, input_backup[j])


        else:
            lutFilePath = os.path.join(lutBasePath, shotLutDic[shot]) + '.cube'
            print "lutFilePath", lutFilePath
            nuke.ViewerProcess.register("SAJA_%s" % shot,
                                        nuke.createNode,
                                        ("Vectorfield",
                                         "vfield_file %s colorspaceIn AlexaV3LogC colorspaceOut linear" % lutFilePath)
                                        )

            allViewer = nuke.allNodes('Viewer')
            if allViewer:
                for i in allViewer:
                    i['viewerProcess'].setValue("SAJA_%s" % shot)
            else:
                nuke.createNode('Viewer').knob('viewerProcess').setValue("SAJA_%s" % shot)
    except:
        pass

def settingPRS():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0.3/config.ocio')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')

def settingSSR():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('SSR')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('SSR')

    nuke.knobDefault("Viewer.viewerProcess", "SSR")
    nuke.root()['format'].setValue('SSR')
    nuke.knobDefault("Root.format", "SSR")
    
    # plate colorspace to alexaV3LogC
    for i in nuke.allNodes('Read'):
        readPath = i['file'].value()
        if ('/show/ssr/' in readPath) and ('/plates/' in readPath):
            i['colorspace'].setValue('AlexaV3LogC')

def settingGCD2():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('sRGB')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('sRGB')

    nuke.knobDefault("Viewer.viewerProcess", 'sRGB')
    nuke.root()['format'].setValue('GCD2')
    nuke.knobDefault("Root.format", "GCD2")

def settingPMC():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('PMC')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('PMC')

    nuke.knobDefault("Viewer.viewerProcess", "PMC")
    nuke.root()['logLut'].setValue('AlexaV3LogC')
    nuke.root()['format'].setValue('PMC')
    nuke.knobDefault("Root.format", "PMC")

def settingDOK_ACES():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('aces_1.0.1')
    #nuke.root().knob('floatLut').setValue('ACES - ACES2065-1')


    # Parsing Sequence name
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')

    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')
    sequence = scsteps[0]

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.709 D60 sim. (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 D60 sim. (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 D60 sim. (ACES)')

def settingTRL():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('TRL')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('TRL')

    format = nuke.toNode('Read1').knob('format').value()

    nuke.knobDefault("Viewer.viewerProcess", "TRL")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue(format)
    nuke.knobDefault("Root.format", "TRL")

def setting1987():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('1987')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('1987')

    nuke.knobDefault("Viewer.viewerProcess", "1987")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('"1987"')
    nuke.knobDefault("Root.format", '"1987"')

def settingGCD1_ZMX():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('sRGB')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('sRGB')

    nuke.knobDefault("Viewer.viewerProcess", "sRGB")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('GCD1')
    nuke.knobDefault("Root.format", "GCD1")

def settingGCD1():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('GCD1')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('GCD1')

    nuke.knobDefault("Viewer.viewerProcess", "GCD1")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('GCD1')
    nuke.knobDefault("Root.format", "GCD1")

def settingMKK3():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Parsing Sequence name
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')

    # set useGPUIfAvailable Expression
    try:
        nukeAllNode = nuke.allNodes()
        for n in nukeAllNode:
            n.knob('useGPUIfAvailable').setExpression('$gui')
    except:
        pass

    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')
    sequence = scsteps[0]

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('MKK3')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('MKK3')

    nuke.knobDefault("Viewer.viewerProcess", "MKK3")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['int16Lut'].setValue('Cineon')
    nuke.root()['format'].setValue('MKK3')
    nuke.knobDefault("Root.format", "MKK3")

def settingMKK3_ACES():
    ## ACES - OCIO project Setting
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('aces_1.0.1')
    nuke.root().knob('workingSpaceLUT').setValue('ACES - ACEScg')
    nuke.root().knob('monitorLut').setValue('ACES/sRGB D60 sim.')
    nuke.root().knob('int8Lut').setValue('Output - Rec.2020')
    nuke.root().knob('int16Lut').setValue('Input - RED - Curve - REDlogFilm')
    nuke.root().knob('logLut').setValue('Input - RED - Curve - REDlogFilm')
    nuke.root().knob('floatLut').setValue('ACES - ACEScg')


    # Parsing Sequence name
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')

    # set useGPUIfAvailable Expression
    try:
        nukeAllNode = nuke.allNodes()
        for n in nukeAllNode:
            n.knob('useGPUIfAvailable').setExpression('$gui')
    except:
        pass

    filename = os.path.basename(fullPath)
    scriptname = os.path.splitext(filename)
    scsteps = scriptname[0].split('_')
    sequence = scsteps[0]

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('Rec.2020 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.2020 (ACES)')

    nuke.knobDefault("Viewer.viewerProcess", 'Rec.2020 (ACES)')
    nuke.root()['format'].setValue('MKK3')
    nuke.knobDefault("Root.format", "MKK3")

def settingGOD():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('GOD')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('GOD')

    nuke.knobDefault("Viewer.viewerProcess", "GOD")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('GOD')
    nuke.knobDefault("Root.format", "GOD")

def settingNJJL():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('NJJL')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('NJJL')

    nuke.knobDefault("Viewer.viewerProcess", "NJJL")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('2K_DCP')
    nuke.knobDefault("Root.format", "2K_DCP")

def settingREAL():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')
    nuke.knobDefault("Viewer.viewerProcess", "AlexaV3Rec709")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('REAL')
    nuke.knobDefault("Root.format", "REAL")

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('sRGB')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('sRGB')

def settingRES():
    # Setup OCIO
    nuke.root().knob('defaultViewerLUT').setValue('OCIO LUTs')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0/config.ocio')
    #nuke.show(nuke.root())
    #nuke.root().forceValidate()
    # Setup Viewer Process    
    nuke.selectAll()
    allNodes = nuke.selectedNodes()
    if allNodes:
        for node in allNodes:
            if node.Class() == 'Viewer':
                node['viewerProcess'].setValue('Rec.709 (ACES)')
            if not nuke.selectedNodes('Viewer'):
                nuke.invertSelection()
                nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    
    nuke.knobDefault("Viewer.viewerProcess", "Rec.709 (ACES)")
    nuke.root()['floatLut'].setValue('linear')

def settingSSSS():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')    

    # Setup Viewer Process    
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i[ 'viewerProcess' ].setValue('SSSS')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('SSSS')
         
    nuke.knobDefault("Viewer.viewerProcess", "SSSS")
    nuke.root()['logLut'].setValue('linear')

def settingKFYG():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')    

    # Setup Viewer Process    
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i[ 'viewerProcess' ].setValue('KFYG')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('KFYG')
         
    nuke.knobDefault("Viewer.viewerProcess", "KFYG")
    nuke.root()['logLut'].setValue('Cineon')
    nuke.root()['format'].setValue('2K_DCP')
    nuke.knobDefault("Root.format", "2K_DCP")

def settingLOG():
    # Setup OCIO
    nuke.root().knob('defaultViewerLUT').setValue('OCIO LUTs')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0/config.ocio')
    #nuke.show(nuke.root())
    #nuke.root().forceValidate()
    # Setup Viewer Process    
    nuke.selectAll()
    allNodes = nuke.selectedNodes()
    if allNodes:
        for node in allNodes:
            if node.Class() == 'Viewer':
                node['viewerProcess'].setValue('Rec.709 (ACES)')
            if not nuke.selectedNodes('Viewer'):
                nuke.invertSelection()
                nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    
    nuke.knobDefault("Viewer.viewerProcess", "Rec.709 (ACES)")
    nuke.root()['floatLut'].setValue('linear')
    #nuke.root()['format'].setValue('LOG')
    #nuke.knobDefault("Root.format", "LOG")

def settingXYFY():
    # Setup OCIO
    nuke.root().knob('defaultViewerLUT').setValue('OCIO LUTs')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue('/backstage/apps/OpenColorIO-Configs/aces_1.0/config.ocio')
    
    nuke.selectAll()
    allNodes = nuke.selectedNodes()
    if allNodes:
        for node in allNodes:
            if node.Class() == 'Viewer':
                node['viewerProcess'].setValue('Rec.709 (ACES)')
            if not nuke.selectedNodes('Viewer'):
                nuke.invertSelection()
                nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    
    nuke.knobDefault("Viewer.viewerProcess", "Rec.709 (ACES)")
    nuke.root()['floatLut'].setValue('linear')
    
    nuke.root().knob('views').fromScript( '\n'.join( ('left #ff0000', 'right #00ff00') ) )
    nuke.root().knob('views_colours').setValue('True')     
    
    nuke.root()['format'].setValue('TISF')
    nuke.knobDefault("Root.format", "TISF")

def settingMKK2():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')    

    # Setup Viewer Process    
    nuke.selectAll()
    allNodes = nuke.selectedNodes()
    if allNodes:
        for node in allNodes:
            if node.Class() == 'Viewer':
                node[ 'viewerProcess' ].setValue('rec709')
            if not nuke.selectedNodes('Viewer'):
                nuke.invertSelection()
                nuke.createNode('Viewer').knob('viewerProcess').setValue('rec709')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('rec709')     
    nuke.knobDefault("Viewer.viewerProcess", "rec709")
    nuke.root()['floatLut'].setValue('cineon2')
            
def setting1953():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')  
        
    # Setup Viewer Process    
    nuke.selectAll()
    allNodes = nuke.selectedNodes()
    if allNodes:
        for node in allNodes:
            if node.Class() == 'Viewer':
                node[ 'viewerProcess' ].setValue('AlexaV3Rec709')
            if not nuke.selectedNodes('Viewer'):
                nuke.invertSelection()
                nuke.createNode('Viewer').knob('viewerProcess').setValue('AlexaV3Rec709')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('AlexaV3Rec709')
    
    # LOG DEFAULT SETTING
    nuke.root()['logLut'].setValue('AlexaV3LogC')
    
    nuke.knobDefault("Viewer.viewerProcess", "AlexaV3LogC")
    
def settingNukeDefault():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')    

    # Setup Viewer Process    
    nuke.selectAll()
    allNodes = nuke.selectedNodes()
    if allNodes:
        for node in allNodes:
            if node.Class() == 'Viewer':
                node[ 'viewerProcess' ].setValue('sRGB')
            if not nuke.selectedNodes('Viewer'):
                nuke.invertSelection()
                nuke.createNode('Viewer').knob('viewerProcess').setValue('sRGB')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('sRGB')
             
    nuke.knobDefault("Viewer.viewerProcess", "sRGB")  
    nuke.root()['floatLut'].setValue('linear')



