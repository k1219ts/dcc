from Katana import NodegraphAPI, Nodes3DAPI

def doIt(node, create=False):
    location = node.getParameter('location').getValue(0)
    variantSetName = node.getParameter('args.variantSetName.value').getValue(0)

    root = Nodes3DAPI.GetGeometryProducer(node)
    loc  = root.getProducerByPath(location)
    attr = loc.getAttribute('info.usd.variants')
    childInfo = attr.getChildByName(variantSetName)
    if not childInfo:
        return

    vals = attr.getChildByName(variantSetName).getData()
    vals = list(vals)
    vals.sort()

    selectParam = node.getParameter('args.variantSelection.value')
    if selectParam.isExpression():
        expression = selectParam.getExpression()
        varName = expression.split('.')[-2]
        SetGraphStateVariables(varName, vals)
    else:
        ifever = SetGraphStateVariables(variantSetName, vals, create)
        if ifever:
            node.getParameter('args.variantSelection.enable').setValue(1, 0)
            node.getParameter('args.variantSelection.value').setExpression(
                'project.variables.%s.value' % variantSetName, True
            )


def SetGraphStateVariables(variantName, values, create=False):
    variablesGroup = NodegraphAPI.GetRootNode().getParameter('variables')
    variableParam  = variablesGroup.getChild(variantName)
    if variableParam:
        optionParam = variableParam.getChild('options')
        optionParam.resizeArray(len(values))
        for i in range(len(values)):
            optionParam.getChildByIndex(i).setValue(values[i], 0)
    else:
        if create:
            variableParam = variablesGroup.createChildGroup(variantName)
            variableParam.createChildNumber('enable', 1)
            variableParam.createChildString('value', values[-1])
            optionParam = variableParam.createChildStringArray('options', len(values))
            for optionParam, optionValue in zip(optionParam.getChildren(), values):
                optionParam.setValue(optionValue, 0)
            return True
