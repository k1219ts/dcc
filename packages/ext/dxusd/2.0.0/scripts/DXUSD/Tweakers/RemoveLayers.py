#coding:utf-8
from __future__ import print_function
import os

from pxr import Sdf

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg
from .Tweaker import Tweaker, ATweaker

class ARemoveLayers(ATweaker):
    def __init__(self, **kwargs):
        ATweaker.__init__(self, **kwargs)

    def Treat(self):
        if not self.inputs:
            msg.warning('No inputs')
            return var.IGNORE
        return var.SUCCESS


class RemoveLayers(Tweaker):
    ARGCLASS = ARemoveLayers
    def DoIt(self):
        for srclyr in self.arg.inputs:
            if isinstance(srclyr, Sdf.Layer):
                srcpath = srclyr.realPath

            del srclyr
            dirpath = utl.DirName(srcpath)
            os.remove(srcpath)

            # 만약 source layer 폴더가 비어있으면, 폴더 삭제 한다.
            try:
                for f in os.listdir(dirpath):
                    if f.startswith('.'):
                        os.remove(utl.SJoin(dirpath, f))
                    else:
                        break
                else:
                    os.rmdir(dirpath)
            except:
                msg.warning('Failed to remove directory (%s)'%dirpath)

        return var.SUCCESS
