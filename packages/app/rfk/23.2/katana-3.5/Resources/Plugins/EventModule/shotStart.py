from Katana import NodegraphAPI, Nodes3DAPI

def firstSetup(param):
    shotName = param.getValue(0)

    variantNodes = NodegraphAPI.GetAllNodesByType('PxrUsdInVariantSelect')
    variantNodes+= NodegraphAPI.GetAllNodesByType('UsdInVariantSelect')
    for node in variantNodes:
    # for node in NodegraphAPI.GetAllNodesByType('PxrUsdInVariantSelect'):
        variantSetName  = node.getParameter('args.variantSetName.value').getValue(0)
        variantSelection= node.getParameter('args.variantSelection.value').getValue(0)
        if variantSetName == 'shotVariant' and variantSelection == shotName:

            location = node.getParameter('location').getValue(0)

            mainProducer = Nodes3DAPI.GetGeometryProducer(node)
            producer = mainProducer.getProducerByPath(location)

            # FrameRange setup
            inTimeAttr = producer.getAttribute('userProperties.inTime')
            outTimeAttr= producer.getAttribute('userProperties.outTime')
            if inTimeAttr and outTimeAttr:
                inTime = inTimeAttr.getValue(0)
                outTime= outTimeAttr.getValue(0)
                NodegraphAPI.SetOutTime(outTime)
                NodegraphAPI.SetInTime(inTime)
                print '[INFO - %s FrameRange]:' % shotName, inTime, outTime
