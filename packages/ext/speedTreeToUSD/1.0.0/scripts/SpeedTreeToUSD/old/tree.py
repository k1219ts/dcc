#!/usr/bin/python

import os
import argparse
import json
import pprint

import hou
import _alembic_hom_extensions as abc


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('abcpath',
                        type=str,
                        help='Alembic file path')
    parser.add_argument('usdpath',
                        type=str,
                        help='JSON file path')
    parser.add_argument('loopRange',
                        type=str, default='1_100',
                        help='Set loopRange of looping (default is 1-100f)')
    parser.add_argument('fps',
                        type=float, default=24.0,
                        help='Set loopRange of looping (default is 1-100f)')
    parser.add_argument('-s', '--hipsave',
                        action='store_true',
                        help='Save hip file.')

    args = parser.parse_args()
    return args

class abcInfo:
    def __init__(self, _abc, _usd, _loopRange, _fps):
        self._abc = _abc
        self._usd = _usd

        print _abc
        print _usd
        print _loopRange
        print _fps

        dir = os.path.dirname(_usd)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        abcRange = abc.alembicTimeRange(_abc)
        self.fps = _fps
        self.start = abcRange[0] * self.fps
        self.end = abcRange[1] * self.fps
        tmp = _loopRange.split('_')
        self.loopRange = (float(tmp[0]), float(tmp[1]))
        self.shellid = []

        self.json = readJsonToDict(self, _abc.replace('abc', 'json'))
        for groupName, prims in self.json.items():
            self.groupName = groupName
            self.prims = {}

            for prim, value in prims.items():
                print prim, value['originalName'], value['sID']
                self.prims['/'+value['originalName']] = prim
                if value['sID']:
                    self.shellid.append('/'+value['originalName'])
        self.set()

    def set(self):
        # check abc file
        try:
            if not os.path.exists(self._abc): raise()
            if not os.path.splitext(self._abc)[1] == '.abc': raise()
        except:
            print 'Error : given file path is not available'
            exit()

    def fullpath(self):
        return os.path.abspath(self._abc)

    def basename(self):
        return os.path.basename(self._abc).split('.')[0]

    def dirname(self):
        return os.path.dirname(self.fullpath())

def readJsonToDict(self, path):
    if os.path.isfile(path):
        f = open(path, "r")
        j = json.load(f)
        f.close()
        return j

def doit(abc, isStatic, save):
    print '-' * 70
    print 'Make Alembic Loop\n'

    # open tmp file
    hip = '{}.hip'.format('.'.join(__file__.split('.')[:-1]))
    hou.hipFile.load(hip)

    print 'hipFile:', hip

    # get nodes
    geo = '/obj/geo1'
    abcImport = hou.node('{}/import_alembic'.format(geo))
    assetName = hou.node('{}/set_originalpath_with_path_and_groups'.format(geo))
    isStaticSwitch = hou.node('{}/isStatic_switch'.format(geo))
    makeLoop = hou.node('{}/make_loop'.format(geo))
    needShellId = hou.node('{}/select_shellid_group'.format(geo))
    renamePrimpath = hou.node('{}/setUsdPrimpath'.format(geo))
    usdOutput = hou.node('{}/usd_export'.format(geo))

    # set modelGroupName
    assetName.setParms({'assetname': abc.groupName})

    # set parameters
    abcImport.setParms({'fileName':abc.fullpath(), 'fps':abc.fps})

    idx = []
    renamePrim = []

    # rename primPath, set SellID
    orgName = abc.prims.keys()
    geom = needShellId.geometry()
    for index, prim in enumerate(geom.prims()):
        for attrib in geom.primAttribs():

            for name in orgName:
                if name == prim.attribValue(attrib):
                    renamePrim.append(abc.prims[name])

            for sID in abc.shellid:
                if sID == prim.attribValue(attrib):
                    print "needshellID: ", index, prim.attribValue(attrib)   # 2 /pine1_Level_1_Cap_PLY
                    if not str(index) in idx:
                            idx.append(str(index))
    print 'idx:', ' '.join(idx)

    renamePrimpath.setParms({'primpaths': ' '.join(renamePrim)})
    needShellId.setParms({'basegroup': '%s' % ' '.join(idx)})

    usdFile = '{}/{}.usd'.format(abc.dirname(), abc.groupName)
    if isStatic:
        usdOutput.setParms({'trange':0, 'usdfile':usdFile})
    else:
        makeLoop.setParms({'start_frame':abc.loopRange[0], 'end_frame':abc.loopRange[1]+1})
        usdOutput.setParms({'f1':abc.loopRange[0]-1, 'f2':abc.loopRange[1]+1, 'usdfile':abc._usd})

    # render the USD
    usdOutput.render(verbose=True, output_progress=True)

    if save:
        hipfile = '{}/{}.hip'.format(abc.dirname(), abc.basename())
        hou.hipFile.save(hipfile)
        print '-' * 70
        print 'Save Hip File :', hipfile


def run():
    print __file__
    args = getArgs()

    abc = abcInfo(args.abcpath, args.usdpath, args.loopRange, args.fps)

    # find static mesh
    isStatic = abc.start == 0 and abc.end == 0

    print '-' * 70
    print 'Alembic File Path :', abc.fullpath()
    if not isStatic:
        print 'Alembic Start Frame :', abc.start
        print 'Alembic End Frame :', abc.end
    else:
        print 'Static Mesh'
    print 'Loop Duration :', abc.loopRange

    # doit(abc, isStatic, args.hipsave)

    print '-' * 70
    print 'export complate!!!'
    exit()


if __name__ == '__main__':
    run()
