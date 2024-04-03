import nuke

def frameAppendClip():
    readNodes = nuke.selectedNodes()
    acNode = nuke.createNode("AppendClip")
    pythonFrameCommand = """[knob firstFrame] - [knob lastFrame]
[python -exec {
theNode = nuke.thisNode()
dots = theNode.dependencies()
for i in range(len(dots)):
  if dots[i].Class() != 'Dot':
    dot = nuke.nodes.Dot()
    dot.setInput(0, theNode.input(i))
    theNode.setInput(i, dot)
    dot['note_font_size'].setValue(20)
    dot['note_font_color'].setValue(16711935)
    dot.setYpos(dot.ypos() + 100)
    dots[i] = dot
  if i == 0:
    dots[i]['label'].setValue(str(theNode.firstFrame()) + '-' + str(dots[i].input(0).lastFrame() - dots[i].input(0).firstFrame() + 1))
  else:
    offset = int(dots[i-1]['label'].value().split('-')[-1])
    dots[i]['label'].setValue(str(offset+1) + '-' + str(offset + dots[i].input(0).lastFrame() - dots[i].input(0).firstFrame() + 1))
  dots[i]['label'].evaluate()
}]"""
    acNode["label"].setValue(pythonFrameCommand)
    if readNodes != []:
        acNode.setYpos(acNode.ypos() + 200)
    acNode["label"].evaluate()