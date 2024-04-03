#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   DEXTER STUDiOS
#
#   CG Supervisor	: seonku.kim
#
#   2017. 11. 13
#-------------------------------------------------------------------------------

import os
import sys
import glob
import re


def get_split_name(f):
    spname = ''
    r = re.compile( r'\.(\d+)\.' )
    match = r.search(f)

    if match:
        spname = match.groups()[0]

    name = f.split( spname )[0]
    ext  = f.split( spname )[-1]
    padding = spname

    return name, ext, padding


def cleanup(path=None, st=1001, et=1002, step=1, pattern='.lock'):
    print '=' * 100
    print ' Cleanup Starting : Delete .lock'
    print '-' * 100

    st = int(float(st))
    et = int(float(et))
    step = int(float(step))

    # 삭제할 폴더에 모든 파일 목록 구하기
    files = filter( os.path.isfile, glob.glob('%s*' % path) )
    files.sort()

    num = 0
    for file in files:
        filename = os.path.basename(file) # Zhuhai_11.1001.iff.lock

        # .lock 파일인 경우만 처리
        if file.endswith(pattern): #.lock
            filename = filename.split(pattern)[0]

            # 1001 패딩 넘버 구하기
            name, ext, padding = get_split_name(filename) # WandaCity_CamB. .exr 1413

            # 해당 구간의 .lock 파일 삭제
            if int(padding) in range(st, et+1, step):
                if os.path.exists( file ):
                    try:
                        os.remove( file )
                        print '%s --> %s' % (file, 'Removed..!!')
                        num += 1
                    except Exception:
                        print '<<WARNNING>> %s' % str(e)


    print ' Result --> %s file(s) is automaticaly removed.' % num
    print '=' * 100


###############################################################################
if __name__ == "__main__":
    if len(sys.argv):
        path = sys.argv[1]
        st = sys.argv[2]
        et = sys.argv[3]
        step = sys.argv[4]
        pattern = sys.argv[5]

        cleanup(path=path, st=st, et=et, step=step, pattern=pattern)

# test
# cleanup(path='/home/seonku.kim/output/test/Zhuhai_11*', st=1001, et=1003, step=1, pattern='.lock')








