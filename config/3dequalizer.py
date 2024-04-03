#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    pkg = '3dequalizer'
    if args.equalizer:
        pkg += '-' + args.equalizer
    result.append(pkg)

    output = ' '.join(result)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
-------------------------------
Dexter Rez 3DEqualizer Launcher
-------------------------------
''',
    )
    parser.add_argument('-v', '--equalizer', type=str, default='4.7',
        help='3DEqualizer version')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
