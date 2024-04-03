#!/usr/bin/python2.7

import os
import sys
import subprocess
import argparse


def main(filename, newer):
    isexts = ['.exr', '.hdr']

    dirname = os.path.dirname(filename)
    basename= os.path.basename(filename)
    basename, ext = os.path.splitext(basename)

    if not ext in isexts:
        return

    _skip = False
    if ext == '.exr':
        p = subprocess.Popen('txinfo %s' % filename, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, errors = p.communicate()
        if 'PixarTextureFormat' in output:
            _skip = True

    # Command
    cmd = 'txmake'
    if newer:
        cmd += ' -newer'
    cmd += ' -envlatl -filter gaussian -format openexr -compression zip -float {INPUT} {OUTPUT}'

    lin_filename = os.path.join(dirname, '%s.exr' % basename)

    # convert Pixar OpenExr TextureFormat
    if not _skip:
        pipe = subprocess.Popen(cmd.format(INPUT=filename, OUTPUT=lin_filename), shell=True)
        pipe.wait()

    _skip = False
    acg_filename = os.path.join(dirname, '%s_acescg.exr' % basename)
    if os.path.exists(acg_filename) and newer:
        _skip = True
        lin_time = os.path.getmtime(lin_filename)
        acg_time = os.path.getmtime(acg_filename)
        if acg_time < lin_time:
            _skip = False

    # convert ACEScg Pixar OpenExr TextureFormat
    if not _skip:
        import ice
        import ACEScgConvert

        loadImg = ice.Load(filename)
        result  = ACEScgConvert.licRec709ToLinAP1(loadImg)
        result.Save(acg_filename, ice.constants.FMT_EXRFLOAT)

        cmd = 'txmake -envlatl -filter gaussian -format openexr -compression zip -float {INPUT} {OUTPUT}'
        pipe = subprocess.Popen(cmd.format(INPUT=acg_filename, OUTPUT=acg_filename), shell=True)
        pipe.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='txenvlatl', description='HDR convert to pixar environment exr')
    parser.add_argument('inputfile', type=str, help='hdr filename')
    parser.add_argument('-newer', action='store_true', help='Do nothing if source is older than target texture')
    args = parser.parse_args()
    main(args.inputfile, args.newer)
