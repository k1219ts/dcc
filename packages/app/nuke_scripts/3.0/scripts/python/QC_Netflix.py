###code 진행중###
import os
import nuke
import DXRulebook.Interface as rb
import nukeCommon as comm



def doit():
    spt_env = os.environ['REZ_NUKE_SCRIPTS_BASE']
    spt = spt_env + '/scripts/toolset/QC_Netflix.nk'
    nuke.loadToolset(os.environ['REZ_NUKE_SCRIPTS_BASE'] + '/scripts/toolset/QC_Netflix.nk')
    # nuke.scriptReadFile(spt)

    # pngPath
    # pngPath   = spt_env + '/scripts/icons/QC_Netflix/'
    # differ    = pngPath + 'QC-Difference.png'
    # grain     = pngPath + 'QC-GrainCheck.png'
    # bright    = pngPath + 'QC-CheckBright.png'
    # dark      = pngPath + 'QC-CheckDark.png'
    #
    gz = nuke.toNode('QC_Netflix')
    #
    # # 그룹 내부 노드
    # Q_differ  = gz.node('Read2')
    # Q_grain   = gz.node('Read3')
    # Q_bright  = gz.node('Read1')
    # Q_dark    = gz.node('Read4')
    #
    # Q_differ['file'].setValue(differ)
    # Q_grain['file'].setValue(grain)
    # Q_bright['file'].setValue(bright)
    # Q_dark['file'].setValue(dark)

    w= nuke.createNode('Write', inpanel=True)
    w.knob('file_type').setValue("jpeg")
    w.knob('_jpeg_quality').setValue(1)

    # read show _config
    configData = comm.getDxConfig()

    if configData['colorSpace'].get('ACES'):
        w.knob('colorspace').setValue('Output - Rec.709')
    elif configData['colorSpace'].get('in')=='Cineon':
        w.knob('colorspace').setValue('rec709')
    else:
        w.knob('colorspace').setValue('Output - sRGB')
        pass

### fullPath ##################
    fullPath = nuke.value('root.name') # /show/7esc/works/CMP/MUD/MUD_0050/precomp/MUD_0050_precomp_v001.nk

### rootName ##################
    coder   = rb.Coder()
    argv    = coder.D.NUKE.WORKS.Decode(os.path.dirname(fullPath))
    argv2   = coder.F.NUKE.Decode(os.path.basename(fullPath))
    argv.update(argv2)
    rootName = os.path.join(coder.D.NUKE.IMAGES.Encode(**argv),'comp')
    #print (rootName)                  # /show/7esc/_2d/shot/MUD/MUD_0050/comp

### version ##################
    file1    = os.path.basename(fullPath) # MUD_0050_precomp_v001.nk
    file2    = os.path.splitext(file1)    # ('MUD_0050_precomp_v001', '.nk')
    file3    = file2[0].split('_')         # ['MUD', '0050', 'precomp', 'v001']
    version = file3[-1]                    # v001


### stat #####################
    splitPath   = fullPath.split('/')
    stat        = splitPath[-2]               # precomp
    techCheck_ver = argv2['seq']+'_'+argv2['shot']+'_techCheck_'+ version     # techCheck_v001

### fileName ##################
    fileName = techCheck_ver+'.%04d'+'.jpg'         # MUD_0050_techCheck_v001.%04d.jpg

### finPath ##################
    finPath = os.path.join(rootName,stat,'images','jpg',techCheck_ver,fileName)
# /show/7esc/_2d/shot/MUD/MUD_0080/comp/precomp/images/jpg/precomp_techCheck_v001/precomp_techCheck_v001.%04d.jpg

    w.knob('file').setValue(finPath)
    w.setInput(0,gz)
