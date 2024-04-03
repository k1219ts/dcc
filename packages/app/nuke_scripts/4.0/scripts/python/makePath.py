


import nuke


def path_sticky():


    mov = nuke.selectedNode().knob('file').value()

    gif = mov.replace('mov','gif')



    pa = 'DCC rez-env ffmpeg-4.2.0 -- ffmpeg -i '
    pb = ' -pix_fmt rgb24 '

    allPath = pa+mov+pb+gif
    a = nuke.createNode('StickyNote')
    a.knob('label').setValue(allPath )
    a.knob('note_font').setValue("DejaVu Sans Bold")
    a.knob('note_font_size').setValue(50)







def createNode():
    cn = nuke.createNode('NoOp')
    cn.setName('makePath')
    ck = nuke.PyScript_Knob('mm','make')
    cn.addKnob(ck)
    setVal = cn.knob('mm')
    setVal.setValue('makePath.path_sticky()')
    cn.knob('tile_color').setValue(3646424319)
    cn.knob('label').setValue('-\nmade by sb')
    cn.knob('note_font').setValue('Bitstream Vera Sans Italic')
