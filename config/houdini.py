#!/bin/python

import os, sys
import argparse

def main(args):
    result = []
    if args.bundle:
        result.append('houBundle-' + args.bundle)

    output = ' '.join(result)
    return output


if __name__ == '__main__':
    epilog_msg = ''
    if '--help' in sys.argv:
        firstpkg = os.getenv('REZ_PACKAGES_PATH').split(':')[0]
        # Project Config
        if '/show/' in firstpkg:
            epilog_msg += '''
----------------------------------
Add Project Config :
    %s''' %  firstpkg

        # Bundle List
        if os.getenv('HOUBUNDLE_PATH'):
            epilog_msg += '''
----------------------------------
         Houdini Bundles
----------------------------------
'''
            hpkgDir = os.getenv('HOUBUNDLE_PATH')
            for i in os.listdir(hpkgDir):
                if os.path.isdir(os.path.join(hpkgDir, i)):
                    epilog_msg += '\t' + i + '\n'
        epilog_msg += ' '

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
----------------------------------
Dexter Rez Houdini Bundle Launcher
----------------------------------
''',
        epilog=epilog_msg
    )
    parser.add_argument('-b', '--bundle', type=str,
        help='Houdini bundle name')

    parser.add_argument('--terminal', action='store_true',
        help='Execute terminal mode')

    args, unknown = parser.parse_known_args()
    result = main(args)
    if not args.terminal:
        if unknown:
            result += ' -- ' + ' '.join(unknown)
    sys.exit(result)
