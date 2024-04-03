#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    pkg = 'rv'
    if args.rv:
        pkg += '-' + args.rv
    result.append(pkg)

    output = ' '.join(result)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
----------------------
Dexter Rez RV Launcher
----------------------
''',
    )
    parser.add_argument('-v', '--rv', type=str, default='1.0.0',
        help='RV version')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)

    if 'REZ_CENTOS_MAJOR_VERSION' in os.environ and '7' == os.environ['REZ_CENTOS_MAJOR_VERSION']:
        result = 'otio-0.13.2 ' + result

    sys.exit(result)
