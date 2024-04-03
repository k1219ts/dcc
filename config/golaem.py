#!/bin/python

import os, sys
import argparse

def main(args):
    # result = []
    result = ['maya_animation', 'maya_asset', 'maya_layout', 'maya_matchmove', 'maya_rigging', 'maya_toolkit', 'dxusd_maya', 'usd_maya']
    pkg = 'golaem_maya'
    if args.golaem:
        pkg += '-' + args.golaem
    result.append(pkg)

    output = ' '.join(result)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
-----------------------------------
Dexter Rez golaem for Maya Launcher
-----------------------------------
''',
    )
    parser.add_argument('-v', '--golaem', type=str, default='8.1.3',
        help='golaem for Maya version')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
