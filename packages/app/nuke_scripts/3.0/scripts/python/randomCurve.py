import random
import nuke

def makeCurve(node):  
    startFrame = node['startFrame'].value()
    endFrame = node['endFrame'].value()
    interval = node['interval'].value()
    
    minValue = node['minValue'].value()
    maxValue = node['maxValue'].value()
    
    curveCount = 0
    for i in node.allKnobs():
        if i.name().startswith('curve'):
            curveCount += 1    
    
    graph = nuke.Double_Knob('curve'+str(curveCount+1), 'curve' + str(curveCount+1) + ' ')
    graph.setRange(minValue, maxValue)
    node.addKnob(graph)

    graph.clearAnimated()
    graph.setAnimated()
    prevValue = None
    for i in range(endFrame):
        if random.uniform(0, interval) <= 1:
            #value = random.randint(minValue, maxValue)
            value = random.uniform(minValue, maxValue)
            
            while value == prevValue:
                value = random.randint(minValue, maxValue)
            prevValue = value
            graph.setValueAt(value, i)
        else:
            pass


def makeNoOp():
    bodyNode = nuke.createNode('NoOp')
    bodyNode.setName('JiHyung Curve')
    
    startFrame = nuke.Int_Knob('startFrame', 'Start Frame: ')
    startFrame.setValue(nuke.Root().firstFrame())
    
    endFrame = nuke.Int_Knob('endFrame', 'End Frame: ')
    endFrame.setValue(nuke.Root().lastFrame())
    
    minValue = nuke.Int_Knob('minValue', 'Minimum Value: ')
    minValue.setValue(0)
    
    maxValue = nuke.Int_Knob('maxValue', 'Maxmum Value: ')
    maxValue.setValue(6)
    
    interval = nuke.Int_Knob('interval', 'Interval: ')
    interval.setValue(1)
    
    makeButton = nuke.PyScript_Knob("makeGraph", 'Make!', 'randomCurve.makeCurve(nuke.thisNode())')  

    bodyNode.addKnob(startFrame)
    bodyNode.addKnob(endFrame)
    bodyNode.addKnob(minValue)
    bodyNode.addKnob(maxValue)
    bodyNode.addKnob(interval)
    makeButton.setFlag(nuke.STARTLINE)
    bodyNode.addKnob(makeButton)

