def registerCurvesUtils():
    from Katana import Nodes3DAPI
    from Katana import FnAttribute

    def buildCurvesUtilsOpChain(node, interface):
        frameTime = interface.getGraphState().getTime()

        interface.setMinRequiredInputs(1)

        argsGb = FnAttribute.GroupBuilder()

        celParam = node.getParameter('CEL')
        if celParam:
            argsGb.set('CEL', celParam.getValue(frameTime))

        rootWidthScaleParam = node.getParameter('rootWidthScale')
        if rootWidthScaleParam:
            argsGb.set('rootWidthScale', rootWidthScaleParam.getValue(frameTime))

        tipWidthScaleParam = node.getParameter('tipWidthScale')
        if tipWidthScaleParam:
            argsGb.set('tipWidthScale', tipWidthScaleParam.getValue(frameTime))

        curvesRatioParam = node.getParameter('curvesRatio')
        if curvesRatioParam:
            argsGb.set('curvesRatio', curvesRatioParam.getValue(frameTime))

        interface.appendOp('CurvesUtils', argsGb.build())

    nodeTypeBuilder = Nodes3DAPI.NodeTypeBuilder('CurvesUtils')

    nodeTypeBuilder.setInputPortNames(('in', ))

    gb = FnAttribute.GroupBuilder()
    gb.set('CEL', FnAttribute.StringAttribute(''))
    gb.set('rootWidthScale', FnAttribute.FloatAttribute(1.0))
    gb.set('tipWidthScale', FnAttribute.FloatAttribute(1.0))
    gb.set('curvesRatio', FnAttribute.FloatAttribute(0.5))

    nodeTypeBuilder.setParametersTemplateAttr(gb.build())
    nodeTypeBuilder.setHintsForParameter('CEL', {'widget': 'cel'})
    nodeTypeBuilder.setHintsForParameter('curvesRatio', {'slider': True})
    nodeTypeBuilder.setBuildOpChainFnc(buildCurvesUtilsOpChain)
    nodeTypeBuilder.build()

registerCurvesUtils()
