#!/bin/python

import os, sys
import argparse

def main(args):
    result = []

    mari_pkg = 'mari'
    if args.mari:
        mari_pkg += '-' + args.mari
        # usd_mari
        if args.mari == '4.6.4':
            result.append('usd_mari')
    result.append(mari_pkg)

    if args.rfm:
        rfm_pkg = 'rfmari'
    if args.rfm != 'rtue':
        if args.mari.startswith('6'):
            args.rfm = '25.2'
        rfm_pkg += '-' + args.rfm
    result.append(rfm_pkg)

    output = ' '.join(result)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
----------------------------------
Dexter Rez Mari RenderMan Launcher
----------------------------------
''',
        epilog='If not arguments, default packages\n\t>> mari-4.6.1 rfmari-23.3\n '
    )
    parser.add_argument('-v', '--mari', type=str, default='4.6.1',
        help='Mari version (default: 4.6.1)')
    parser.add_argument('-r', '--rfm', type=str, default='23.3',
        help='RenderManForMari version (default: 23.3)')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if unknown:
        if args.terminal:
            if len(unknown) > 1:
                result += ' -- ' + ' '.join(unknown[1:])
        else:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
