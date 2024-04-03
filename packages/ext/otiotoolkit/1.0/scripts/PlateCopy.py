#coding:utf-8
import os, argparse, sys

def plateCopy(args):
    if not os.path.exists(os.path.dirname(args.dstFileRule)):
        os.makedirs(os.path.dirname(args.dstFileRule))

    for index, frame in enumerate(range(args.frameIn, args.frameIn + args.duration + 1)):
        srcFile = args.srcFileRule % (int(args.srcStartFrame) + index)
        dstFile = args.dstFileRule % frame
        cmd = 'cp -rvf %s %s' % (srcFile, dstFile)
        os.system(cmd)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # mov file name
    argparser.add_argument('-ssf', '--srcStartFrame', dest='srcStartFrame', type=int, help='plate start frame number')
    argparser.add_argument('-sfr', '--srcFileRule', dest='srcFileRule', type=str, default='', help='plate file rule')
    argparser.add_argument('-dfr', '--dstFileRule', dest='dstFileRule', type=str, default='', help='rename : dst file rule')
    argparser.add_argument('-d', '--duration', dest='duration', type=int, help='scan duration')
    argparser.add_argument('-fi', '--frameIn', dest='frameIn', type=int, help='working frame IN')

    args, unknown = argparser.parse_known_args(sys.argv)

    plateCopy(args)