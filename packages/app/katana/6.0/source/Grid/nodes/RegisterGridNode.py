
def registerGrid():
    from Katana import Nodes3DAPI
    from Katana import FnAttribute

    def buildGridOpChain(node, interface):
        # Get the current frame time
        frameTime = interface.getGraphState().getTime()

        # Set the minimum number of input ports
        interface.setMinRequiredInputs(0)

        argsGb = FnAttribute.GroupBuilder()

        # Parse node parameters
        locationParam = node.getParameter('location')

        # Add the Grid Op to the chain
        interface.appendOp('Grid', argsGb.build())


    # Create a NodeTypeBuilder to register the new type
    nodeTypeBuilder = Nodes3DAPI.NodeTypeBuilder('Grid')

    # Build the node's parameters
    gb = FnAttribute.GroupBuilder()
    gb.set('location', FnAttribute.StringAttribute('/root/world/geo/grid1'))

    # Set the parameters template
    nodeTypeBuilder.setParametersTemplateAttr(gb.build())

    # Set parameter hints
    nodeTypeBuilder.setHintsForParameter('location', {'widget': 'scenegraphLocation'})

    # Set the callback responsible to build the Ops chain
    nodeTypeBuilder.setBuildOpChainFnc(buildGridOpChain)

    # Build the new node type
    nodeTypeBuilder.build()

# Register the node
registerGrid()
