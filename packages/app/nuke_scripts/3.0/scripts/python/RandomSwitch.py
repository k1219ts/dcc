import random
import nuke


def createRandomSwitch():

    sn = nuke.createNode('Switch')
    time = nuke.Int_Knob('tt','time')
    sn.addKnob(time)
    inPut = nuke.Int_Knob('ii','input')
    sn.addKnob(inPut)
    sn.setName('randomSwich')
    setii = sn.knob('ii')
    setii.setExpression('[python {nuke.thisNode().inputs()}]')
    val = sn.knob('which')
    val.setExpression('int(random(frame-(tt))*(int(ii)))')
    sn.knob('tile_color').setValue(183052543)
    sn.knob('label').setValue('-\nmade by sb')
    sn.knob('note_font').setValue('Bitstream Vera Sans Italic')

