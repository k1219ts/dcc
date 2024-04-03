import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

import shutil

if __name__ == '__main__':
    orgpath = '/show/pipe/_3d/shot/RUN/RUN_0190/layout/sphere/v005/sphere_layout.geom.org.usd'
    testpath = '/show/pipe/_3d/shot/RUN/RUN_0190/layout/sphere/v005/sphere_layout.geom.usd'

    shutil.copyfile(orgpath, testpath)

    arg = twk.APreReference()
    arg.inputs = [testpath]

    if arg.Treat() == var.SUCCESS:
        print(arg)

        t = twk.PreReference(arg)
        t.DoIt()
