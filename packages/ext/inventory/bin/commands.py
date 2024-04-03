import os
import sys

commands = sys.argv[1]
dirpath = sys.argv[2]

print commands, dirpath
for directory in os.listdir(dirpath):
    if commands == "gif_make":
        cmd = 'python /WORK_DATA/develop/dcc/packages/ext/inventory/bin/gif_make.py --src %s' % os.path.join(dirpath, directory)
        os.system(cmd)
    elif commands == 'thumb_make':
        cmd = 'python /WORK_DATA/develop/dcc/packages/ext/inventory/bin/thumb_make.py --directory %s' % os.path.join(dirpath, directory)
        os.system(cmd)
    elif commands == 'into_db':
        cmd = 'DCC.local dev python-2 -- python /WORK_DATA/develop/dcc/packages/ext/inventory/bin/into_db.py %s' % os.path.join(dirpath, directory)
        os.system(cmd)