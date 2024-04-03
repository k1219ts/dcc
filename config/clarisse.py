#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    clarisse_pkg = 'clarisse'
    if args.clarisse:
        clarisse_pkg += '-' + args.clarisse
    result.append(clarisse_pkg)

    output = ' '.join(result)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
----------------------------
Dexter Rez Clarisse Launcher
----------------------------
'''
    )
    parser.add_argument('-v', '--clarisse', type=str,
        help='Clarisse version')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
