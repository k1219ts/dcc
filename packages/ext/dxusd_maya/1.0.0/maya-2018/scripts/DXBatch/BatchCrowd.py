import os, argparse, datetime, sys, getpass

# Tractor
import TractorConfig

import DXUSD_MAYA.Crowd as Crowd
import DXUSD_MAYA.MUtils as mutl
import DXRulebook.Interface as rb


def doIt(args):
    import maya.cmds as cmds

    plugins  = ['pxrUsd', 'pxrUsdTranslators', 'AbcExport', 'DXUSD_Maya', 'backstageMenu']

    isGolaem = False
    isMiarmy = False

    requests = os.environ['REZ_USED_REQUEST'].split()
    for r in requests:
        if r.startswith('golaem_maya'):
            isGolaem = True
            plugins.append('glmCrowd')
        if r.startswith('miarmy'):
            isMiarmy = True
            plugins.append('MiarmyProForMaya2018')

    mutl.InitPlugins(plugins)

    # file load
    cmds.file(args.file, f=True, o=True)
    cmds.select(cl=True)

    coder = rb.Coder()

    ret = coder.D.Decode(args.outDir)

    # showName = ret.show
    # shotName = '{SEQ}_{SHOT}'.format(SEQ=ret.seq, SHOT=ret.shot)

    # coll = DBConnection('WORK', showName)

    # dbItem = {'action': 'export', 'filepath': args.file, 'user': args.user, 'time': datetime.datetime.now().isoformat(),
    #           'shot':shotName, 'task':'', 'enabled': False}

    version, glmCaches = args.crowd[0].split('=')

    if isMiarmy:
        Crowd.shotExport_miarmy(show=ret.show, seq=ret.seq, shot=ret.shot, version=version,
                                fr=args.frameRange, efr=args.exportRange, user=args.user, process=args.process)
    if isGolaem:
        Crowd.shotExport_golaem(show=ret.show, seq=ret.seq, shot=ret.shot, version=version,
                                fr=args.frameRange, efr=args.exportRange, glmCaches=glmCaches, user=args.user, process=args.process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DXUSD_MAYA Crowd Miarmy Batch script.'
    )

    parser.add_argument('-f', '--file', type=str, help='Maya filename.')
    parser.add_argument('-o', '--outDir', type=str, help='Cache out directory.')
    parser.add_argument('-u', '--user', type=str, help='User name')

    # TimeRange argument
    parser.add_argument('-fr',  '--frameRange', type=int, nargs=2, default=[0, 0], help='frame range, (start, end)')
    parser.add_argument('-efr', '--exportRange', type=int, nargs=2, default=[0, 0], help='export frame range, (start, end)')
    parser.add_argument('-fs',  '--frameSample', type=float, default=1.0, help='frame step size default = 1.0')

    # Acting argument
    parser.add_argument('-p', '--process', type=str, choices=['both', 'geom', 'comp'], help='task export when choice process, [geom, comp]')
    parser.add_argument('-hs', '--host', type=str, choices=['local', 'spool', 'tractor'], help='if host local, cache export. other option is "spool" spool')

    #   Crowd Out
    parser.add_argument('-cw', '--crowd', type=str, nargs='*',
                        help='export nodeName of dxCamera \nex) --crowd v005=dxCamera1 v004=dxCamera2')

    args, unknown = parser.parse_known_args(sys.argv)
    if not args.file:
        assert False, 'not found file.'

    from pymel.all import *

    doIt(args)
