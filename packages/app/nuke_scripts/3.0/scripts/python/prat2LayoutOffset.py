import dxConfig
from pymongo import MongoClient

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'

def getOffsetRange(show, shot):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[show]

    camData = coll.find({'data_type':'camera', 'task': 'matchmove', 'show': show, 'shot': shot}).sort('time', -1)
    distData = coll.find({'data_type':'distortion', 'show': show, 'shot': shot}).sort('time', -1)

    for i in camData:
        if 'layout' in i['task_publish']['plateType']:
            layStart = int(i['task_publish']['startFrame'])
    for i in distData:
        plateStart = int(i['task_publish']['startFrame'])

    if layStart or plateStart:
        offset = layStart - plateStart
        print('offset: %s - %s = %s' % (layStart, plateStart, offset))
        return offset
    else:
        print('not offset!')
        return False

# getOffsetRange('prat2', 'PS59_0200')
