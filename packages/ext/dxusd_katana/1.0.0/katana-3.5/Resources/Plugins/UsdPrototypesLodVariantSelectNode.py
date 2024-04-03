def registerProtoVariantSelect():
    from Katana import Nodes3DAPI, NodegraphAPI
    from Katana import FnAttribute, FnGeolibServices

    def buildOpChain(self, interface):
        interface.setExplicitInputRequestsEnabled(True)

        graphState = interface.getGraphState()
        frameTime  = interface.getFrameTime()
        location   = self.getParameter('location').getValue(frameTime)

        variantSelection = None
        if self.getParameter('args.variantSelection.enable').getValue(frameTime):
            variantSelection = self.getParameter('args.variantSelection.value').getValue(frameTime)
        #
        if location and variantSelection is not None:
            valueAttr = FnAttribute.StringAttribute(variantSelection)
            gb = FnAttribute.GroupBuilder()
            _ifever = 0
            # Prototypes
            for locParam in self.getParameter('PrototypeLocations').getChildren():
                location = locParam.getValue(frameTime)
                if location:
                    _ifever += 1
                    entryName = FnAttribute.DelimiterEncode(location)
                    entryPath = 'variants.' + entryName + '.lodVariant'
                    gb.set(entryPath, valueAttr)

            if _ifever > 0:
                existingValue = (interface.getGraphState().getDynamicEntry('var:pxrUsdInSession'))
                if isinstance(existingValue, FnAttribute.GroupAttribute):
                    gb.deppUpdate(existingValue)

                graphState = (
                    graphState.edit()
                    .setDynamicEntry('var:pxrUsdInSession', gb.build())
                    .build()
                )

        interface.addInputRequest('in', graphState)


    nb = Nodes3DAPI.NodeTypeBuilder('UsdPrototypesLodSelect')
    nb.setInputPortNames(('in',))

    # Parameter
    pgb = FnAttribute.GroupBuilder()
    pgb.set('location', '')
    pgb.set('args.variantSelection', '')
    pgb.set('PrototypeLocations', FnAttribute.StringAttribute([]))
    nb.setParametersTemplateAttr(pgb.build())

    nb.setHintsForParameter('location', {'widget': 'scenegraphLocation'})
    nb.setHintsForParameter('args.variantSelection', {'widget': 'popup', 'options': ['high', 'mid', 'low'], 'editable': 1})
    nb.setHintsForParameter('PrototypeLocations', {'widget': 'scenegraphLocationArray'})

    nb.setGenericAssignRoots('args', '__variantUI')

    nb.setBuildOpChainFnc(buildOpChain)

    nb.build()


registerProtoVariantSelect()
