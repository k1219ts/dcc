import hou

def ratioSizeUpdate():
    sourceIndiciesNode = hou.pwd()
    sourceNode = '/'.join(sourceIndiciesNode.path().split('/')[:-1]) + '/sourceWeight'
    connectionNode = hou.node(sourceNode)
    size = connectionNode.parm('values').evalAsInt()

    for i in range(size):
        connectionNode.parm('value%d' % i).set(i)
        # connectionNode.parm.setExpression('../sourceIndices/ratio%d' % i)
        connectionNode.parm('weight%d' % i).setExpression("ch('../sourceIndicies/ratio%d')" % (i + 1))
