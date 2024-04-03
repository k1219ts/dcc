#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    if not args.kat and not args.rfk and not args.usd:
        result.append('katana-4')
        result.append('rfk-23.5')
        result.append('usd_katana-19.11')
    else:
        kat_pkg = 'katana'
        if args.kat:
            kat_pkg += '-' + args.kat
        result.append(kat_pkg)
        rfk_pkg = 'rfk'
        if args.rfk:
            rfk_pkg += '-' + args.rfk
        result.append(rfk_pkg)

        if 'katana' in kat_pkg:
		    usd_pkg = 'usd_katana'
		    if args.usd:
		        usd_pkg += '-' + args.usd
		    result.append(usd_pkg)

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
    epilog_msg += 'If not arguments, default packages\n\t>> katana-3.5 rfk-23.3 usd_katana-19.11\n '

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
--------------------------------------
Dexter Rez RenderManForKatana Launcher
--------------------------------------
''',
        epilog=epilog_msg
    )
    parser.add_argument('-k', '--kat', type=str,
        help='Katana version. If not specified, resolved by other options.')
    parser.add_argument('-r', '--rfk', type=str,
        help='RenderManForKatana version.  If not specified, resolved by other options.')
    parser.add_argument('-u', '--usd', type=str,
        help='USD version.  If not specified, resolved by other options.')

    parser.add_argument('--golaem', type=str, nargs='?', default=None, const='true',
        help='Golaem version (recommend: 7.2.2)')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    # print args, unknown
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
