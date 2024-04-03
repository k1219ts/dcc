from Katana import NodegraphAPI, Utils


def parameter_setValue_Event(eventType, eventID, node, param):

    if node == NodegraphAPI.GetRootNode():
        # shot framerange setup support
        if param.getFullName() == 'rootNode.variables.shotVariant.value':
            import EventModule.shotStart as shotStart
            shotStart.firstSetup(param)
    else:
        ntype = node.getType()

        # PointInstancer prototypes lodVariant select - PrototypeLocation setup support
        if ntype == 'UsdPrototypesLodSelect':
            if param.getName() == 'location':
                import EventModule.protoTypes as protoTypes
                protoTypes.SetPrototypesLocation(node)



# register
Utils.EventModule.RegisterEventHandler(parameter_setValue_Event, eventType='parameter_setValue')
