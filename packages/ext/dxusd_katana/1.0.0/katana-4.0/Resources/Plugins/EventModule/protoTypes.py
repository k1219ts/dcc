from Katana import NodegraphAPI, Nodes3DAPI

def SetPrototypesLocation(node):
    location = node.getParameter('location').getValue(0)
    locParam = node.getParameter('PrototypeLocations')

    rootProducer = Nodes3DAPI.GetGeometryProducer(node)

    producer = rootProducer.getProducerByPath(location)
    if producer.getType() != 'usd point instancer':
        locParam.resizeArray(0)
        return

    prototypes = list()

    producer = rootProducer.getProducerByPath(location + '/Prototypes')
    for p in producer.iterChildren():
        prototypes.append(p.getFullName())

    locParam.resizeArray(len(prototypes))
    params = locParam.getChildren()
    for i in range(len(params)):
        params[i].setValue(prototypes[i], 0)
