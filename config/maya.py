#!/bin/python

import os, sys
import argparse

def main(args):
    result = ['maya_animation', 'maya_asset', 'maya_layout', 'maya_matchmove', 'maya_rigging', 'maya_toolkit', 'dxusd_maya']
    if not args.maya and not args.usd:
        result.append('maya-2018')
        result.append('usd_maya-19.11')
    else:
        maya_pkg = 'maya'
        if args.maya:
            maya_pkg += '-' + args.maya
        result.append(maya_pkg)
        if args.maya:
            usd_pkg = 'usd_maya'
        if args.usd:
            usd_pkg += '-' + args.usd
        if args.usd != 'false':
            result.append(usd_pkg)

    if args.ziva:
        ziva_pkg = 'ziva'
        if args.ziva != 'true':
            ziva_pkg += '-' + args.ziva
        result.append(ziva_pkg)
    if args.golaem:
        golaem_pkg = 'golaem_maya-8.1.3'
        if args.golaem != 'true':
            golaem_pkg += '-' + args.golaem
        result.append(golaem_pkg)
    if args.miarmy:
        miarmy_pkg = 'miarmy'
        if args.miarmy != 'true':
            miarmy_pkg += '-' + args.miarmy
        result.append(miarmy_pkg)
    if args.rfm:
        rfm_pkg = 'rfm'
        if args.rfm != 'true':
            rfm_pkg += '-' + args.rfm
        result.append(rfm_pkg)
    if args.mtoa:
        mtoa_pkg = 'mtoa'
        if args.mtoa != 'true':
            mtoa_pkg += '-' + args.mtoa
        result.append(mtoa_pkg)
    if args.redshift:
        redshift_pkg = 'redshift'
        if args.redshift != 'true':
            redshift_pkg += '-' + args.redshift
        result.append(redshift_pkg)
    if args.beyondscreen:
        beyondscreen_pkg = 'beyondscreen'
        if args.beyondscreen != 'true':
            beyondscreen_pkg += '-' + args.beyondscreen
        result.append(beyondscreen_pkg)
    if args.vray:
        vray_pkg = 'vray'
        if args.vray != 'true':
            vray_pkg += '-' + args.vray
        result.append(vray_pkg)

    # will be delete
    if args.zelos:
        result.append('zelos')
    if args.tane:
        result.append('tane')
        if not 'zelos' in result:
            result.append('zelos')
    if args.bora:
        result.append('bora')

    output = ' '.join(result)
    return output

if __name__ == '__main__':
    extra = []
    if '-h' in sys.argv:
        sys.argv.remove('-h')
        extra.append('-h')
    epilog_msg = ''
    if '--help' in sys.argv:
        # Project Config
        if os.getenv('PROJECTCONFIG'):
            epilog_msg += '''
Add Project Config:
    %s

''' % os.getenv('PROJECTCONFIG')
    epilog_msg += 'If not arguments, default packages\n\t>> maya-2018 usd_maya-19.11\n '

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
----------------------------
Dexter Rez Maya USD Launcher
----------------------------
''',
        epilog=epilog_msg
    )
    parser.add_argument('-v', '--maya', type=str,
        help='Maya version. If not specified, resolved by other options.')
    parser.add_argument('-u', '--usd', type=str,
        help='USD version. If not specified, resolved by other options.')

    parser.add_argument('--ziva', type=str, nargs='?', default=None, const='true',
        help='ZivaVFX version (recommend: 1.7)')
    parser.add_argument('--golaem', type=str, nargs='?', default=None, const='true',
        help='Golaem version (recommend: 7.3.9)')
    parser.add_argument('--miarmy', type=str, nargs='?', default=None, const='true',
        help='Miarmy version (recommend: 6.5.21)')
    parser.add_argument('--rfm', type=str, nargs='?', default=None, const='true',
        help='RenderManForMaya version (recommend: 23.2)')
    parser.add_argument('--mtoa', type=str, nargs='?', default=None, const='true',
        help='RenderManForMaya version (recommend: 4.1.1)')
    parser.add_argument('--redshift', type=str, nargs='?', default=None, const='true',
        help='Redshift version (recommend: 2.5.52)')
    parser.add_argument('--beyondscreen', type=str, nargs='?', default=None, const='true',
        help='BeyondScreen version 1.0')
    parser.add_argument('--vray', type=str, nargs='?', default=None, const='true',
        help='V-ray version (recommend: 3.60.01)')

    # will be delete
    parser.add_argument('--zelos', action='store_true',
        help='Zelos. Only for Maya2018')
    parser.add_argument('--tane', action='store_true',
        help='Tane. Only for Maya2018')
    parser.add_argument('--bora', action='store_true',
        help='Bora. Only for Maya2018')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)

    if unknown:
        if args.terminal:
            if len(unknown) > 1:
                unknown += extra
                result += ' -- ' + ' '.join(unknown[1:])
        else:
            unknown += extra
            result += ' -- ' + ' '.join(unknown)

    sys.exit(result)
