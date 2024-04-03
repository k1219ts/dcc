#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    mbd_pkg = 'motionbuilder'
    if args.motionbuilder:
        mbd_pkg += '-' + args.motionbuilder
    result.append(mbd_pkg)

    output = ' '.join(result)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
---------------------------------
Dexter Rez MotionBuilder Launcher
---------------------------------
''',
    )
    parser.add_argument('-v', '--motionbuilder', type=str,
        help='MotionBuilder version')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
