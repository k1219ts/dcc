from Katana import NodegraphAPI, Utils

def paramSetEvent(eventType, eventID, node, param):
    if node == NodegraphAPI.GetRootNode():
        # shot FrameRange setup
        if param.getFullName() == 'rootNode.variables.shot.value':
            import EventModule.shotStart as shotStart
            shotStart.firstSetup(param)

    else:
        ntype = node.getType()

        # PointInstancer prototypes lodVariant select - PrototypeLocation setup
        if ntype == 'UsdPrototypesLodSelect':
            if param.getName() == 'location':
                import EventModule.protoTypes as protoTypes
                protoTypes.SetPrototypesLocation(node)

# register
Utils.EventModule.RegisterEventHandler(paramSetEvent, eventType='parameter_setValue')
