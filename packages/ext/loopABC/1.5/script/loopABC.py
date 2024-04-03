#!/usr/bin/python

'''
Make sequenced alembic loop using houdini

'''


import os, sys
import argparse, alembic


HOUDINI_VERSION = os.environ['HOUDINI_VERSION']
HFS = '%s/houdini/python2.7libs'%os.environ['HFS']



def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fps',
                        type=float, default=24.0,
                        help='Set fps (default is 24.0)')
    parser.add_argument('-s', '--hipsave',
                        action='store_true',
                        help='Save hip file.')
    parser.add_argument('-c', '--checkABC',
                        action='store_true',
                        help='Check frame loopRange of given alembic.')
    parser.add_argument('loopRange',
                        type=int, default=(1, 100), nargs='*',
                        help='Set loopRange of looping (default is 1-100f)')
    parser.add_argument('abcpath',
                        type=str,
                        help='Alembic file path')

    args = parser.parse_args()
    if len(args.loopRange) == 1:
        args.loopRange = [1, args.loopRange[0]]

    if args.loopRange[0] > args.loopRange[1]:
        print 'Error : %s' % 'loopRange is not available'
        exit()

    return args


class abcInfo:
    def __init__(self, _abc, fps, loopRange, check):
        self._abc = _abc

        self.fps = fps
        self.loopRange = loopRange
        self.start = None
        self.end = None

        self.set()
        self.setFrameloopRange()

    def set(self):
        # check abc file
        try:
            if not os.path.exists(self._abc): raise()
            if not os.path.splitext(self._abc)[1] == '.abc': raise()
        except:
            print 'Error : given file path is not available'
            exit()


    def findFirstMesh(self, obj):
        md = obj.getMetaData()
        if alembic.AbcGeom.IPolyMesh.matches(md):
            return obj

        for child in obj.children:
            child = self.findFirstMesh(child)
            if child:
                return child


    def setFrameloopRange(self):
        iarch = alembic.Abc.IArchive(str(self.fullpath()))
        xform = alembic.AbcGeom.IPolyMesh(self.findFirstMesh(iarch.getTop()),
                                     alembic.Abc.WrapExistingFlag.kWrapExisting)
        schema = xform.getSchema()
        ts = schema.getTimeSampling()
        tsType = schema.getTimeSampling().getTimeSamplingType()
        numTimeSample = schema.getNumSamples()

        minTime = ts.getSampleTime(0)
        maxTime = ts.getSampleTime(numTimeSample - 1)
        stepSize = float(tsType.getTimePerCycle())

        self.start = minTime / stepSize
        self.end = maxTime / stepSize
        self.fps = 1 / stepSize


    def fullpath(self):
        return os.path.abspath(self._abc)

    def basename(self):
        return os.path.basename(self._abc).split('.')[0]

    def dirname(self):
        return os.path.dirname(self.fullpath())


def houdiniRun(doit):
    if hasattr(sys, "setdlopenflags"):
        old_dlopen_flags = sys.getdlopenflags()
        import DLFCN
        sys.setdlopenflags(old_dlopen_flags | DLFCN.RTLD_GLOBAL)

    print '-' * 70
    print 'Run Houdini ({})\n'.format(HOUDINI_VERSION)

    try:
        sys.path.append(HFS)
        import hou
    except ImportError:
        print 'Error : hou import error'
        return False
    finally:
        if hasattr(sys, "setdlopenflags"):
            sys.setdlopenflags(old_dlopen_flags)

    doit()

    hou.exit()
    print '-' * 70
    print 'Houdini Exited'

    return True

def doit(abc, isStatic, save):
    print '-' * 70
    print 'Make Alembic Loop\n'

    # open tmp file
    hip = '{}/loopABC.hip'.format(os.environ['LOOPABC_SOURCEPATH'])
    hou.hipFile.load(hip)


    # get nodes
    geo = '/obj/alembic1'
    abcImport = hou.node('{}/alembic1'.format(geo))
    abcLoop   = hou.node('{}/make_loop'.format(geo))
    abcOutput = hou.node('{}/rop_alembic1'.format(geo))
    isStaticSwitch = hou.node('{}/isStatic_switch'.format(geo))

    # set parameters
    abcImport.setParms({'fileName':abc.fullpath(), 'fps':abc.fps})

    if isStatic:
        abcOutput.setParms({'trange':0,
                            'filename':'{}/{}_static.abc'.format(abc.dirname(), abc.basename())})
    else:
        abcLoop.setParms({'start_frame':abc.loopRange[0], 'end_frame':abc.loopRange[1]+1})
        abcOutput.setParms({'f1':abc.loopRange[0]-1, 'f2':abc.loopRange[1]+1,
                            'filename':'{}/{}_loop.abc'.format(abc.dirname(), abc.basename())})

    # render the alembic
    abcOutput.render(verbose=True, output_progress=True)

    if save:
        hipfile = '{}/{}.hip'.format(abc.dirname(), abc.basename())
        hou.hipFile.save(hipfile)
        print '-' * 70
        print 'Save Hip File :', hipfile


def run():
    print __file__
    args = getArgs()
    abc = abcInfo(args.abcpath, args.fps, args.loopRange, args.checkABC)

    # find static mesh
    isStatic = abc.start == 0 and abc.end == 0


    print '-' * 70
    print 'Alembic File Path :', abc.fullpath()
    if not isStatic:
        print 'Alembic Start Frame :', abc.start
        print 'Alembic End Frame :', abc.end
    else:
        print 'Static Mesh'
    print 'Loop Duration :', args.loopRange

    if not args.checkABC:
        houdiniRun(lambda *f: doit(abc, isStatic, args.hipsave, ))

    exit()


if __name__ == '__main__':
    run()
