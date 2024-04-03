import os

import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg


def getGeomFiles(walkDir):
    result = list()
    for root, dirs, names in os.walk(walkDir):
        for name in names:
            if name.endswith('_geom.usd'):
                result.append(os.path.join(root, name))
    return result


def doIt(rootDir):
    arg = twk.ALoopClip()
    arg.timeScales = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0]
    # arg.timeScales = [1.0]
    arg.loopRange  = (1001, 5000)
    arg.clipRange  = (1001, 1055)
    arg.geomfiles  = getGeomFiles(rootDir)

    if arg.Treat():
        print(arg)
        cliptwk = twk.LoopClip(arg)
        cliptwk.DoIt()


if __name__ == '__main__':
    # rootDir = '/show/prat2/_3d/asset/cow/clip/cow_slow_0030/v002/base'
    # # getGeomFiles(rootDir)
    # doIt(rootDir)

    rootDirs = [
        # '/show/prat2/_3d/asset/cow/clip/cow_slow_0020/v002/base',
        # '/show/prat2/_3d/asset/cow/clip/cow_slow_0030/v002/base',
        # '/show/prat2/_3d/asset/cow/clip/cow_slow_0040/v002/base',
        # '/show/prat2/_3d/asset/cow/clip/cow_slow_0050/v002/base',
        '/show/pipe/_3d/asset/cow/clip/cow_slow_0050/v002/base'
    ]
    for d in rootDirs:
        doIt(d)
