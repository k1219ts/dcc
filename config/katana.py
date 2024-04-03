#!/bin/python

import os, sys
import argparse

def main(args):
    result = []

    kat_pkg = 'katana'
    if args.kat:
        kat_pkg += '-' + args.kat
    result.append(kat_pkg)

    if args.usd:
        usd_pkg = 'usd_katana'
        if args.usd != 'true':
            usd_pkg += '-' + args.usd
        result.append(usd_pkg)
    if args.rfk:
        rfk_pkg = 'rfk'
        if args.rfk != 'true':
            rfk_pkg += '-' + args.rfk
        result.append(rfk_pkg)
    if args.ktoa:
        ktoa_pkg = 'ktoa'
        if args.ktoa != 'true':
            ktoa_pkg += '-' + args.ktoa
        result.append(ktoa_pkg)
    if args.golaem:
        golaem_pkg = 'golaem_katana'
        if args.golaem != 'true':
            golaem_pkg += '-' + args.golaem
        result.append(golaem_pkg)

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
    epilog_msg += 'If not arguments, default packages\n\t>> katana-3.5.2\n '

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
--------------------------
Dexter Rez Katana Launcher
--------------------------
''',
        epilog=epilog_msg
    )
    parser.add_argument('-k', '--kat', type=str,
        help='Katana version. If not specified, resolved by other options.')

    parser.add_argument('-u', '--usd', type=str, nargs='?', default=None, const='true',
        help='USD version (recommend: 19.11)')

    parser.add_argument('--rfk', type=str, nargs='?', default=None, const='true',
        help='RenderManForKatana version (recommend: 23.2)')
    parser.add_argument('--ktoa', type=str, nargs='?', default=None, const='true',
        help='KatanaToArnold version (recommend: 2.4)')

    parser.add_argument('--golaem', type=str, nargs='?', default=None, const='true',
        help='Golaem version (recommend: 7.2)')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
