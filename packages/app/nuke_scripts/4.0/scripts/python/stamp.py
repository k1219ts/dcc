import os
import json
import sys
# import getpass
import nuke
import nukescripts
import compWrite
import stamp_info
from dxConfig import dxConfig
from python import nukeCommon as comm


def stampDEFAULT():
    stampNode = nuke.createNode('stamp_default')

    stampNode['Project_name'].setValue(stamp_info.getPrjName())
    stampNode['Shotname'].setValue(stamp_info.getShotName())
    stampNode['Artist_name'].setValue(stamp_info.getUserName())

    #from configData_letterBox
    configData = comm.getDxConfig()
    if configData :
        ratioData = configData['letterBox']['ratio']

        if ratioData == 'fullgate':
            stampNode['LetterBox'].setValue(0)
        elif ratioData == '2.35:1':
            stampNode['LetterBox'].setValue(1)
        elif ratioData == '2.39:1':
            stampNode['LetterBox'].setValue(2)
        elif ratioData == '1.85:1':
            stampNode['LetterBox'].setValue(3)
        elif ratioData == '2.2:1':
            stampNode['LetterBox'].setValue(4)
        elif ratioData == '2:1':
            stampNode['LetterBox'].setValue(5)
        else :
            stampNode['LetterBox'].setValue(0)

        stampNode['MaskOpacity'].setValue(0.85)
        Mask = configData['letterBox']['Mask']
        stampNode['MaskOpacity'].setValue(float(Mask))

    else:
        stampNode['LetterBox'].setValue(0)
        stampNode['MaskOpacity'].setValue(0.85)

    return stampNode


def stampNetflix():
    stampNode = nuke.createNode('stamp_netflix')

    stampNode['Project_name'].setValue(stamp_info.getPrjName())
    stampNode['Shotname'].setValue(stamp_info.getShotName())
    stampNode['Artist_name'].setValue(stamp_info.getUserName())

    #from configData_letterBox
    configData = comm.getDxConfig()
    if configData :
        ratioData = configData['letterBox']['ratio']

        if ratioData == 'fullgate':
            stampNode['LetterBox'].setValue(0)
        elif ratioData == '2.35:1':
            stampNode['LetterBox'].setValue(1)
        elif ratioData == '2.39:1':
            stampNode['LetterBox'].setValue(2)
        elif ratioData == '1.85:1':
            stampNode['LetterBox'].setValue(3)
        elif ratioData == '2.2:1':
            stampNode['LetterBox'].setValue(4)
        elif ratioData == '2:1':
            stampNode['LetterBox'].setValue(5)
        else :
            stampNode['LetterBox'].setValue(0)

        stampNode['MaskOpacity'].setValue(0.85)
        Mask = configData['letterBox']['Mask']
        stampNode['MaskOpacity'].setValue(float(Mask))

    else:
        stampNode['LetterBox'].setValue(0)
        stampNode['MaskOpacity'].setValue(0.85)

    return stampNode




''' not use

def Stamp(project = None):
    worker = getpass.getuser()

    if project == 'CRF':
        crfStamp = nuke.createNode('crfStamp')
        crfStamp.knob('Artist_name').setValue(worker)
        crfStamp.knob('Shotname').setValue('CRF3')
        crfStamp.knob('Project_name').setValue('CRF3')
        crfStamp.knob('formatsize').setValue(0)
        if not (nuke.root().name()) == '':
            crfStamp.knob('Shotname').setValue('_'.join(os.path.basename(nuke.root().name()).split('_')[:2]))

    elif project == 'MKK':
        crfStamp = nuke.createNode('crfStamp')
        crfStamp.setName('MKK_Stamp')
        crfStamp.knob('Artist_name').setValue(worker)
        crfStamp.knob('Shotname').setValue('FFT2')
        crfStamp.knob('Project_name').setValue(project)
        crfStamp.knob('formatsize').setValue(2)
        crfStamp.knob('custom_res').setValue(2048, 0)
        crfStamp.knob('custom_res').setValue(1024, 1)

        if not (nuke.root().name()) == '':
            crfStamp.knob('Shotname').setValue('_'.join(os.path.basename(nuke.root().name()).split('_')[:2]))
    else:
        nuke.createNode('DDstamp_v4').knob('Artist_name').setValue(worker)

def stampMkk():

    env25dList = ['NZR_0090','NZR_0140','NZR_0160','NZR_0240',
                  'NZR_0290','NZR_0820','NZR_1090'
                  ]

    if 'main' in nuke.root()['views'].toScript():
        stampNode = nuke.createNode('MMK_guide_stamp_single')
    else:
        stampNode = nuke.createNode('MMK_guide_stamp')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
        if shotName in env25dList:
            stampNode['ENV25D'].setValue(True)

        else:
            stampNode['ENV25D'].setValue(False)

def stampPICN():
    stampNode = nuke.createNode('stamp_picn')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)

def stampPRAT():
    stampNode = nuke.createNode('stamp_prat')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)

def stampTISF():
    stampNode = nuke.createNode('stamp_tisf')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampSSY():
    stampNode = nuke.createNode('stamp_ssy')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampSSY_no_stereo():
    stampNode = nuke.createNode('stamp_ssy_no_stereo')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampHYDE():
    stampNode = nuke.createNode('stamp_hyde')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampDKF():
    stampNode = nuke.createNode('stamp_dkf')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampWEST():
    stampNode = nuke.createNode('stamp_1953')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampGBL():
    stampNode = nuke.createNode('stamp_gbl')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    stampNode['tc_offset'].setValue(nuke.root().firstFrame())

    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampMKK2():
    stampNode = nuke.createNode('stamp_mkk2')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode



def stampMKK2_lnr():
    stampNode = nuke.createNode('stamp_mkk2_lnr')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampMKK2_fx():
    stampNode = nuke.createNode('stamp_mkk2_fx')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampLOG():
    import json

    stampNode = nuke.createNode('stamp_log')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
        #/show/log/stuff/shot_reel_json
        reelData = {}
        seq = shotName.split('_')[0]
        if os.path.exists('/show/log/stuff/shot_reel_json/%s_reel.json' % seq.lower()):
            reelData = json.loads(open('/show/log/stuff/shot_reel_json/%s_reel.json' % seq.lower(), 'r').read())
#        if 'PRS' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/prs_reel.json', 'r').read())
#        elif 'SWR' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/swr_reel.json', 'r').read())
#        elif 'TRI' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/tri_reel.json', 'r').read())

        #------------------------------------------------------------------------------
        if reelData.get(shotName):
            reelNames = sorted(reelData[shotName])
            reelCount = len(reelNames)
            reelName = '\n'.join(reelNames)
            print(reelNames)

            stampNode.node('P_INPUT1')['message'].setValue(reelName)
            trvalue = stampNode.node('P_INPUT1')['translate'].value()
            stampNode.node('P_INPUT1')['translate'].setValue((trvalue[0],trvalue[1]+(17*(reelCount-1))))

        else:
            stampNode.node('P_INPUT1')['message'].setValue('')


    return stampNode

def stampLOG_NO_1011():
    import json
    stampNode = nuke.createNode('stamp_log_no_1011')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
        #/show/log/stuff/shot_reel_json
        reelData = {}
        seq = shotName.split('_')[0]
        if os.path.exists('/show/log/stuff/shot_reel_json/%s_reel.json' % seq.lower()):
            reelData = json.loads(open('/show/log/stuff/shot_reel_json/%s_reel.json' % seq.lower(), 'r').read())
#        if 'PRS' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/prs_reel.json', 'r').read())
#        elif 'SWR' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/swr_reel.json', 'r').read())
#        elif 'TRI' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/tri_reel.json', 'r').read())

        #------------------------------------------------------------------------------
        if reelData.get(shotName):
            reelNames = sorted(reelData[shotName])
            reelCount = len(reelNames)
            reelName = '\n'.join(reelNames)
            print(reelNames)

            stampNode.node('P_INPUT1')['message'].setValue(reelName)
            trvalue = stampNode.node('P_INPUT1')['translate'].value()
            stampNode.node('P_INPUT1')['translate'].setValue((trvalue[0],trvalue[1]+(17*(reelCount-1))))

        else:
            stampNode.node('P_INPUT1')['message'].setValue('')
    return stampNode


def stampLOG_slate():
    import json
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter/show/'):
        fullPath = fullPath.replace('/netapp/dexter/show/', '/show/')

    stampNode = nuke.createNode('stamp_log_slate')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (fullPath) == '':
        shotName = '_'.join(os.path.basename(fullPath).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
        #/show/log/stuff/shot_reel_json
        reelData = {}
        seq = shotName.split('_')[0]
        if os.path.exists('/show/log/stuff/shot_reel_json/%s_reel.json' % seq.lower()):
            reelData = json.loads(open('/show/log/stuff/shot_reel_json/%s_reel.json' % seq.lower(), 'r').read())


#        if 'PRS' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/prs_reel.json', 'r').read())
#        elif 'SWR' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/swr_reel.json', 'r').read())
#        elif 'TRI' in shotName:
#            reelData = json.loads(open('/show/log/stuff/shot_reel_json/tri_reel.json', 'r').read())

        #------------------------------------------------------------------------------
        if reelData.get(shotName):
            reelNames = sorted(reelData[shotName])
            reelCount = len(reelNames)
            reelName = '\n'.join(reelNames)
            print(reelNames)

            stampNode.node('P_INPUT1')['message'].setValue(reelName)
            trvalue = stampNode.node('P_INPUT1')['translate'].value()
            stampNode.node('P_INPUT1')['translate'].setValue((trvalue[0],trvalue[1]+(17*(reelCount-1))))

        else:
            stampNode.node('P_INPUT1')['message'].setValue('')
        #------------------------------------------------------------------------------
        stampNode['Team'].setValue(fullPath.split('/')[6])
        stampNode['script_version'].setValue('_'.join(os.path.splitext(os.path.basename(fullPath))[0].split('_')[2:4]))

    return stampNode


def stampNJJL():
    stampNode = nuke.createNode('stamp_njjl')
    stampNode['Project_name'].setValue('NJJL')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampKFYG():
    stampNode = nuke.createNode('stamp_kfyg_hw')
    stampNode['Project_name'].setValue('KFYG')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampXYFY():
    stampNode = nuke.createNode('stamp_xyfy')
    stampNode['Project_name'].setValue('XYFY')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampSSSS():

    stampNode = nuke.createNode('stamp_ssss')

    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)

    return stampNode

def stampSSSS_internal():

    stampNode = nuke.createNode('stamp_ssss_internel')
    stampNode['Project_name'].setValue('SSSS')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        shotName = '_'.join(os.path.basename(nuke.root().name()).split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampRES():
    stampNode = nuke.createNode('stamp_res')
    stampNode['Project_name'].setValue('RES')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampREAL():
    stampNode = nuke.createNode('stamp_real')
    stampNode['Project_name'].setValue('REAL')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode


def slateXYFY():
    slate = nuke.createNode('slate_xyfy')
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        #DZR_0020_keying_v007.nk
        fileEl = nkFileName.split('_')
        seq = fileEl[0]
        shotNumber = fileEl[1]
        version = fileEl[-1].split('.')[0]
        slate['shot'].setValue(seq)
        slate['cutnum'].setValue(shotNumber)
        slate['ver'].setValue('2' + version[1:])
        slate['script'].setValue(nkFileName)
        return slate

def stampXYFY_Delivery():
    stamp = nuke.createNode('stamp_xyfy_roll')
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        #DZR_0020_keying_v007.nk
        fileEl = nkFileName.split('_')
        seq = fileEl[0]
        shotNumber = fileEl[1]
        shotName = seq+'_'+shotNumber

        version = fileEl[-1].split('.')[0]
        if os.path.exists('/show/xyfy/stuff/roll/roll.json'):
            reelData = json.loads(open('/show/xyfy/stuff/roll/roll.json', 'r').read())

        stamp['shot'].setValue(seq)
        stamp['cutnum'].setValue(shotNumber)
        stamp['ver'].setValue('2' + version[1:])
        stamp['script'].setValue(nkFileName.upper())
        if reelData.get(shotName):
            stamp['roll'].setValue(reelData[shotName])

        return stamp

def stampGOD():

    stampNode = nuke.createNode('stamp_god')
    stampNode['Project_name'].setValue('GOD')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode


def stampWKZ():
    #nuke.createNode('wkz_viewer_lut', inpanel = False)
    stampNode = nuke.createNode('stamp_wkz')
    stampNode['Project_name'].setValue('WKZ')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampDRG():
    stampNode = nuke.createNode('stamp_drg')
    stampNode['Project_name'].setValue('DRG')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampMKK3():
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])

        # json import
        reelData = json.loads(open('/stuff/mkk3/stuff/temp/stamp/mkk3_reelname_v02.json', 'r').read())
        shotValue = reelData[shotName]
        clipname = shotValue['clipname']

        if clipname == 'FULL_CG':
            timeCode = nuke.createNode('AddTimeCode', inpanel = False)
            timeCode.knob('startcode').setValue('00:00:00:01')
            timeCode.knob('useFrame').setValue(True)
            timeCode.knob('frame').setValue(float(1001))
        else:
            pass

        stampNode = nuke.createNode('stamp_mkk3', inpanel = False)
        stampNode.knob('Clipname').setValue(clipname)
        stampNode.knob('Shotname').setValue(shotName)

    return stampNode



def stampGCD1():
    nkFilePath = nuke.root().name()

    if nkFilePath.startswith('/netapp/dexter'):
        nkFilePath = nkFilePath.replace('/netapp/dexter', '')
    prj = nkFilePath.split('/')[2]
    prjName = prj.upper()
    #scriptstepsA = nkFilePath.split('/')[-1]
    #scriptstepsB = scriptstepsA.split('.')[0]
    #scriptstepsC = scriptstepsB.split('_')[0:4]
    #script = '_'.join(scriptstepsC) + '.nk'

    stampNode = nuke.createNode('stamp_gcd1')
    stampNode['Project_name'].setValue(prjName)
    #stampNode['Script_name'].setValue(script)
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampNMY():
    stampNode = nuke.createNode('stamp_nmy')
    stampNode['Project_name'].setValue('NMY')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stamp1987():
    stampNode = nuke.createNode('stamp_1987')
    stampNode['Project_name'].setValue('1987')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampGOE():
    stampNode = nuke.createNode('stamp_goe')
    stampNode['Project_name'].setValue('GOE')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampMRM():
    nkFilePath = nuke.root().name()

    if nkFilePath.startswith('/netapp/dexter'):
        nkFilePath = nkFilePath.replace('/netapp/dexter', '')
    prj = nkFilePath.split('/')[2]
    prjName = prj.upper()
    #scriptstepsA = nkFilePath.split('/')[-1]
    #scriptstepsB = scriptstepsA.split('.')[0]
    #scriptstepsC = scriptstepsB.split('_')[0:4]
    #script = '_'.join(scriptstepsC) + '.nk'

    if prj == 'dok' :
        reFormat = nuke.createNode('Reformat')
        reFormat.knob('resize').setValue(0)
        reFormat.knob('format').setValue('DOK')


    stampNode = nuke.createNode('stamp_mrm')
    stampNode['Project_name'].setValue(prjName)
    #stampNode['Script_name'].setValue(script)
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)

    if prj == 'rom7' :
        stampNode['LetterBox'].setValue(2)

    elif prj == 'pmc' :
        stampNode['LetterBox'].setValue(3)

    else :
        stampNode['LetterBox'].setValue(0)

    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

def stampBTL():
    stampNode = nuke.createNode('stamp_btl')
    stampNode['Project_name'].setValue('BTL')
    worker = getpass.getuser()
    stampNode['Artist_name'].setValue(worker)
    if not (nuke.root().name()) == '':
        nkFileName = os.path.basename(nuke.root().name())
        shotName = '_'.join(nkFileName.split('_')[:2])
        stampNode.knob('Shotname').setValue(shotName)
    return stampNode

'''
