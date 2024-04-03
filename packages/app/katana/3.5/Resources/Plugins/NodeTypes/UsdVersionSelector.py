from Katana import (
    Nodes3DAPI,
    NodegraphAPI,
    RenderingAPI,
    FnAttribute,
    FnGeolibServices
)

nb = Nodes3DAPI.NodeTypeBuilder('UsdVersionSelector')
nb.setInputPortNames(('in', ))

nb.setParametersTemplateAttr(FnAttribute.GroupBuilder()
    .set('location', '')
    .set('args.variantSetName', '')
    .set('args.variantSelection', '')
    .set('additionalLocations', FnAttribute.StringAttribute([]))
    .build())

nb.setHintsForParameter('location', {'widget': 'scenegraphLocation'})
nb.setHintsForParameter('additionalLocations', {'widget': 'scenegraphLocationArray'})

nb.setGenericAssignRoots('args', '__variantUI')

def buildOpChain(self, interface):
    interface.setExplicitInputRequestsEnabled(True)

    graphState = interface.getGraphState()

    frameTime = interface.getFrameTime()

    location = self.getParameter("location").getValue(frameTime)

    variantSetName = ''
    if self.getParameter("args.variantSetName.enable").getValue(frameTime):
        variantSetName = self.getParameter("args.variantSetName.value").getValue(
                frameTime)


    variantSelection = None
    if self.getParameter("args.variantSelection.enable").getValue(frameTime):
         variantSelection = self.getParameter(
                "args.variantSelection.value").getValue(frameTime)

    if location and variantSetName and variantSelection is not None:
        entryName = FnAttribute.DelimiterEncode(location)
        entryPath = "variants." + entryName + "." + variantSetName

        valueAttr = FnAttribute.StringAttribute(variantSelection)
        gb = FnAttribute.GroupBuilder()
        gb.set(entryPath, valueAttr)

        for addLocParam in self.getParameter(
                'additionalLocations').getChildren():
            location = addLocParam.getValue(frameTime)
            if location:
                entryName = FnAttribute.DelimiterEncode(location)
                entryPath = "variants." + entryName + "." + variantSetName
                gb.set(entryPath, valueAttr)


        existingValue = (
                interface.getGraphState().getDynamicEntry("var:pxrUsdInSession"))

        if isinstance(existingValue, FnAttribute.GroupAttribute):
            gb.deepUpdate(existingValue)

        graphState = (graphState.edit()
                .setDynamicEntry("var:pxrUsdInSession", gb.build())
                .build())


    interface.addInputRequest("in", graphState)

nb.setBuildOpChainFnc(buildOpChain)

def getScenegraphLocation(self, frameTime):
    location = self.getParameter('location').getValue(frameTime)
    if not (location == '/root' or location.startswith('/root/')):
        location = '/root'
    return location

nb.setGetScenegraphLocationFnc(getScenegraphLocation)


def appendToParametersOpChain(self, interface):
    frameTime = interface.getFrameTime()

    location = self.getScenegraphLocation(frameTime)
    variantSetName = ''
    if self.getParameter('args.variantSetName.enable').getValue(frameTime):
        variantSetName = self.getParameter(
                'args.variantSetName.value').getValue(frameTime)

    # This makes use of the attrs recognized by PxrUsdInUtilExtraHintsDap
    # to provide the hinting from incoming attr values.
    uiscript = '''
        local variantSetName = Interface.GetOpArg('user.variantSetName'):getValue()

        local variantsGroup = (Interface.GetAttr('info.usd.variants') or
                GroupAttribute())

        local variantSetNames = {}
        for i = 0, variantsGroup:getNumberOfChildren() - 1 do
            variantSetNames[#variantSetNames + 1] = variantsGroup:getChildName(i)
        end

        Interface.SetAttr("__pxrUsdInExtraHints." ..
                Attribute.DelimiterEncode("__variantUI.variantSetName"),
                        GroupBuilder()
                            :set('widget', StringAttribute('popup'))
                            :set('options', StringAttribute(variantSetNames))
                            :set('editable', IntAttribute(1))
                            :build())

        local variantOptions = {}

        if variantSetName ~= '' then
            local variantOptionsAttr =
                    variantsGroup:getChildByName(variantSetName)
            if Attribute.IsString(variantOptionsAttr) then
                variantOptions = variantOptionsAttr:getNearestSample(0.0)
            end
        end

        Interface.SetAttr("__pxrUsdInExtraHints." ..
                Attribute.DelimiterEncode("__variantUI.variantSelection"),
                        GroupBuilder()
                            :set('widget', StringAttribute('popup'))
                            :set('options', StringAttribute(variantOptions))
                            :set('editable', IntAttribute(1))
                            :build())
    '''

    sscb = FnGeolibServices.OpArgsBuilders.StaticSceneCreate(True)

    sscb.addSubOpAtLocation(location, 'OpScript.Lua',
            FnAttribute.GroupBuilder()
                .set('script', uiscript)
                .set('user.variantSetName', variantSetName)
                .build())

    interface.appendOp('StaticSceneCreate', sscb.build())


nb.setAppendToParametersOpChainFnc(appendToParametersOpChain)

nb.build()
