#coding:utf-8
from __future__ import print_function

import os, argparse, datetime, sys, traceback, json
import BatchFeather

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DXUSD_MAYA Batch script.'
    )

    parser.add_argument('-i', '--inputCache', type=str, help='input cache')
    parser.add_argument('-g', '--groomFile', type=str, help='groom asset scene or groom simulation scene')
    parser.add_argument('-nv', '--nsver', type=str, help='groom nsver')
    # parser.add_argument('-o', '--outDir', type=str, help='Cache out directory.')
    parser.add_argument('-u', '--user', type=str, help='User name')

    # TimeRange argument
    parser.add_argument('-fr',  '--frameRange', type=int, nargs=2, default=[0, 0], help='frame range, (start, end)')
    # parser.add_argument('-efr', '--exportRange', type=int, nargs=2, default=[0, 0], help='export frame range, (start, end)')
    # parser.add_argument('-fs',  '--frameSample', type=float, default=1.0, help='frame step size default = 1.0')
    parser.add_argument('-s',   '--step', type=float, default=1.0, help='frame step size default = 1.0')

    # Acting argument
    parser.add_argument('-p', '--process', type=str, choices=['both', 'geom', 'comp'], default='both', help='task export when choice process, [geom, comp]')
    parser.add_argument('-hs', '--host', type=str, choices=['local', 'spool', 'tractor'], help='if host local, cache export. other option is "spool" spool')
    parser.add_argument('-pp', '--primPattern', type=str, help='Prim Pattern')
    parser.add_argument('-gu', '--groomUsd', type=str, help='Groom usd file path')

    # task argument
    #   Groom Out
    # parser.add_argument('-g', '--groom', action='count', default=0, help='if groom, export groom after exported mesh')
    # parser.add_argument('-og', '--onlyGroom', action='count', default=0, help='if groom, export groom after exported mesh')
    # parser.add_argument('-gs', '--groomSim', type=str, nargs='*', help='export Groom of already simulation geoCache')

    args, unknown = parser.parse_known_args(sys.argv)
    print('# DEBUG :', args.inputCache)

    if not args.inputCache:
        sys.exit(1)

    BatchFeather.FeatherDefomerSceneSetup(args.inputCache, args.groomFile,
                                          fr=args.frameRange, step=args.step,
                                          primPattern=args.primPattern)

    # if not args.outDir:
    #     flags = rb.Flags(pub='_3d')
    #     flags.D.SetDecode(os.path.dirname(args.file), 'ROOTS')
    #     flags.F.MAYA.SetDecode(os.path.basename(args.file), 'BASE')
    #     args.outDir = flags.D.SHOT

    # if args.host == 'spool':
    #     job = SceneSpool(args).doIt()
    # else:
    #     try:
    #         SceneExport(args)
    #     except Exception as e:
    #         errorStr = traceback.format_exc()
    #
    #         cmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
    #         cmd += ['BotMsg', '--artist', 'daeseok.chae']
    #         cmd += ['--message', '\"%s\"' % errorStr]
    #         cmd += ['--bot', 'BadBot']
    #         # print 'Cmd:', ' '.join(cmd)
    #         os.system(' '.join(cmd))
    #         os._exit(1)

    # quit
    sys.exit(0)
