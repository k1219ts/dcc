
def registerDxUsdFeatherOp():
    """
    Registers a new GeoScaler node type using the NodeTypeBuilder utility
    class.
    """

    from Katana import Nodes3DAPI
    from Katana import FnAttribute

    def buildDxUsdFeatherOpChain(node, interface):
        # Get the current frame time
        f = interface.getGraphState().getTime()

        # Set the minimum number of input ports
        interface.setMinRequiredInputs(1)

        argsGb = FnAttribute.GroupBuilder()

        # Parse node parameters
        CELParam = node.getParameter('CEL')
        if CELParam:
            argsGb.set('CEL', CELParam.getValue(f))

        # purposeParm = node.getParameter('purpose')
        # if purposeParm:
        #     argsGb.set('purpose', purposeParm.getValue(f))

        sWidthMulParm = node.getParameter('stemWidthMultiplier')
        if sWidthMulParm:
            argsGb.set('stemWidthMultiplier', sWidthMulParm.getValue(f))

        bWidthMulParm = node.getParameter('barbWidthMultiplier')
        if bWidthMulParm:
            argsGb.set('barbWidthMultiplier', bWidthMulParm.getValue(f))

        barbProbabilityParm = node.getParameter('barbProbability')
        if barbProbabilityParm:
            argsGb.set('barbProbability', barbProbabilityParm.getValue(f))

        lamProbabilityParm = node.getParameter('laminationProbability')
        if barbProbabilityParm:
            argsGb.set('laminationProbability', lamProbabilityParm.getValue(f))

        # Add the GeoScaler Op to the Ops chain
        interface.appendOp('DxUsdFeatherOp', argsGb.build())

    # Create a NodeTypeBuilder to register the new type
    nodeTypeBuilder = Nodes3DAPI.NodeTypeBuilder('DxUsdFeatherOp')

    # Add input port
    nodeTypeBuilder.setInputPortNames(("in",))

    # Build the node's parameters
    gb = FnAttribute.GroupBuilder()
    gb.set('CEL', FnAttribute.StringAttribute('//**/Groom/Render/*'))
    gb.set('stemWidthMultiplier', FnAttribute.FloatAttribute(1.0))
    gb.set('barbWidthMultiplier', FnAttribute.FloatAttribute(1.0))
    gb.set('barbProbability', FnAttribute.FloatAttribute(1.0))
    gb.set('laminationProbability', FnAttribute.FloatAttribute(1.0))
    # gb.set('purpose', FnAttribute.IntAttribute(0))

    # Set the parameters template
    nodeTypeBuilder.setParametersTemplateAttr(gb.build())

    # Set parameter hints
    nodeTypeBuilder.setHintsForParameter('CEL', {'widget': 'cel'})
    # nodeTypeBuilder.setHintsForParameter('purpose',
    #     {'widget':'mapper', 'options':{'Render':0, 'Proxy':1}}
    # )


    # Set the callback responsible to build the Ops chain
    nodeTypeBuilder.setBuildOpChainFnc(buildDxUsdFeatherOpChain)

    # Build the new node type
    nodeTypeBuilder.build()


# Register the node
registerDxUsdFeatherOp()
