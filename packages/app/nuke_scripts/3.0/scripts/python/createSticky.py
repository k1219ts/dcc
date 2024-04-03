

import nuke


def shot_sticky():

    asd = []
    for node in nuke.selectedNodes():

        nodePath = node['file'].value()
        nodeSp = nodePath.split('/')[-1]
        nodeName = nodeSp.split('_')

        done = nodeName[0]+'_'+nodeName[1]
        asd.append(done)
  
    a = nuke.createNode('StickyNote')
    b= 'name' + '\n'
    tt=[]
    for i in asd:

        tt.append(i)
        tt.append('|')
    tt.pop()

    for i in tt:
        b = b + i



    a.knob('label').setValue(b+ '\n' )
    a.knob('note_font').setValue("DejaVu Sans Bold")
    a.knob('note_font_size').setValue(50)



def createSticky():
    cn = nuke.createNode('NoOp')
    cn.setName('shot_name_make')
    ck = nuke.PyScript_Knob('mm','make')
    cn.addKnob(ck)
    setVal = cn.knob('mm')
    setVal.setValue('createSticky.shot_sticky()')
    cn.knob('tile_color').setValue(3636472319)
    cn.knob('label').setValue('-\nmade by sb')
    cn.knob('note_font').setValue('Bitstream Vera Sans Italic')





