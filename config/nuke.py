#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    if not args.nuke:
        result.append('nuke-12.2.4')
    else:
        nuke_pkg = 'nuke'
        if args.nuke:
            nuke_pkg += '-' + args.nuke
        result.append(nuke_pkg)

    # if args.usd:
    #     result.append('usd_nuke-' + args.usd)
    if args.eddy:
        eddy_pkg = 'eddy'
        if args.eddy != 'true':
            eddy_pkg += '-' + args.eddy
        result.append(eddy_pkg)

    output = ' '.join(result)
    return output


if __name__ == '__main__':
    epilog_msg = ''
    if '--help' in sys.argv:
        # Project Config
        if os.getenv('PROJECTCONFIG'):
            epilog_msg += '''
Add Project Config :
    %s

''' % os.getenv('PROJECTCONFIG')
    epilog_msg += 'If not arguments, default packages\n\t>> nuke-10.0.4\n '

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
------------------------
Dexter Rez Nuke Launcher
------------------------
''',
        epilog=epilog_msg
    )
    parser.add_argument('-v', '--nuke', type=str,
        help='Nuke version. If not specified, resolved by other options.')

    # parser.add_argument('--usd', type=str, nargs='?', default=None, const='19.07',
    #     help='USD version (default: 19.07)')
    parser.add_argument('--eddy', type=str, nargs='?', default=None, const='true',
        help='Eddy version (recommend: 2.4.1)')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
