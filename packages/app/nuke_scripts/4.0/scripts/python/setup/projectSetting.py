import nuke, json
import sys, os

Fps = 23.976
Fps2 = 23.98

def settingDIE():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/nuke_config.ocio') #1.3

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    nuke.knobDefault("Viewer.viewerProcess", "ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES2065-1')
    nuke.root()['format'].setValue('DIE')
    nuke.knobDefault("Root.format", "DIE")
    nuke.root()['fps'].setValue(Fps)

def settingCDH():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/nuke_config.ocio') #1.3

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    nuke.knobDefault("Viewer.viewerProcess", "ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES2065-1')
    nuke.root()['format'].setValue('BDS')
    nuke.knobDefault("Root.format", "BDS")
    nuke.root()['fps'].setValue(Fps2)

def settingEMD():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/nuke_config.ocio') #1.3

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    nuke.knobDefault("Viewer.viewerProcess", "ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES2065-1')
    nuke.root()['format'].setValue('DOK')
    nuke.knobDefault("Root.format", "DOK")
    nuke.root()['fps'].setValue(Fps2)

def settingSLC():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/nuke_config.ocio') #1.3

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    nuke.knobDefault("Viewer.viewerProcess", "ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES2065-1')
    nuke.root()['format'].setValue('DOK')
    nuke.knobDefault("Root.format", "DOK")
    nuke.root()['fps'].setValue(Fps2)

def settingMGD():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/nuke_config.ocio') #1.3

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    nuke.knobDefault("Viewer.viewerProcess", 'MGD')

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('Input - RED - REDLog3G10 - REDWideGamutRGB')
    nuke.root()['format'].setValue('2K_DCP')
    nuke.knobDefault("Root.format", "2K_DCP")


def setting7ESC():
    nuke.root().knob('defaultViewerLUT').setValue('Nuke Root LUTs')
    nuke.root().knob('OCIO_config').setValue('nuke-default')

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue('rec.709')
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue('rec.709')

    nuke.knobDefault("Viewer.viewerProcess", 'rec.709')

    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('Cineon')
    nuke.root()['format'].setValue('2K_DCP')
    nuke.knobDefault("Root.format", "2K_DCP")
    nuke.root()['fps'].setValue(float(23.976))


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


    # def settingPRAT2():
    #     nuke.root().knob('colorManagement').setValue('OCIO')
    #     nuke.root().knob('OCIO_config').setValue('custom')
    #     nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/config.ocio') #1.0.3
    #
    #     # Setup Viewer Process
    #     allViewer = nuke.allNodes('Viewer')
    #     if allViewer:
    #         for i in allViewer:
    #             i['viewerProcess'].setValue('Rec.709 (ACES)')
    #     else:
    #         nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
    #
    #     nuke.knobDefault("Viewer.viewerProcess", 'Rec.709 (ACES)')
    #
    #     allRead = nuke.allNodes('Read')
    #     for i in allRead:
    #         if 'plates' in i['file'].value():
    #             i['colorspace'].setValue('ACES - ACES2065-1')
    #     nuke.root()['format'].setValue('BDS')
    #     nuke.knobDefault("Root.format", "BDS")
    #     nuke.root()['fps'].setValue(Fps)
