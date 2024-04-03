#coding:utf-8
from __future__ import print_function
import inspect

from pxr import Sdf

from DXUSD.Structures import Arguments, Layers
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg


class ATweaker(Arguments):
    def __init__(self, **kwargs):
        self.inputs  = Layers()
        self.outputs = Layers()

        # search path
        self.searchPath = list()

        Arguments.__init__(self, **kwargs)

    def GetSearchPath(self):
        paths = []

        if self.show:
            paths.append(self.D.PUB)
            paths.extend(utl.GetSharedDirs(self.D.OTHER.CONFIG))
        if self.seq:
            paths.insert(0, self.D.SEQ)
        if self.shot:
            paths.insert(0, self.D.SHOT)

        # elif self.customdir:
        #     paths.append(self.customdir)
        if self.customdir:
            if self.customdir.startswith('/assetlib'):
                paths.append(self.customdir)

        return paths


class AMasterPack(ATweaker):
    def __init__(self, **kwargs):
        '''
        dstdir (str): target dirpath
        master (str): package output name
        geomfiles (list[str]) : geometry usd files
        '''
        self.master = None
        self.dstdir = None
        self.geomfiles = []

        ATweaker.__init__(self, **kwargs)


class Tweaker(object):
    '''
    All tweakers are inherited from this class
    '''
    ARGCLASS = ATweaker

    def __init__(self, arg):
        self.__name__ = self.__class__.__name__
        self.arg = self.ARGCLASS(**arg.AsDict())
        # if arg.__name__ == self.ARGCLASS.__name__:
        #     self.arg = arg
        # else:
        #     self.arg = self.ARGCLASS(**arg.AsDict())

    def copy(self):
        arg = self.ARGCLASS(**self.arg.AsDict())
        return self.__class__(arg)

    def DoIt(self):
        msg.error('doIt@Tweaker : Must override doIt methode.')


class LazyTweaker:
    '''

    '''
    def __init__(self, tweaker, args):
        '''
        '''
        self.__name__ = self.__class__.__name__

        self.tweaker = tweaker
        self.args    = args

        if not isinstance(self.args, list):
            self.args = [args]


class Tweak(object):
    def __init__(self):
        self.__name__ = self.__class__.__name__
        self.queue = list()


    def __lshift__(self, other):
        self.Add(other)


    def Add(self, tweaker, index=None):
        if str(inspect.getmro(type(tweaker))[1]) == str(Tweaker) or \
           isinstance(tweaker, LazyTweaker):
            if index == None:
                self.queue.append(tweaker)
            else:
                self.queue.insert(index, tweaker)
        elif str(inspect.getmro(type(tweaker))[0]) == str(Tweak):
            self.queue.extend(tweaker.queue)
        else:
            msg.warning('add@Tweak : ',
                        'Given arguemnt(%s) is not Tweaker'%str(tweaker),
                        dev=True)

    def DoIt(self, save=True):
        # Start Tweaking
        msg.debug('-'*70)
        msg.debug('[ Start Tweaking ]')

        for q in self.queue:
            tweakers = []
            if isinstance(q, LazyTweaker):
                for arg in q.args:
                    tweakers.append(q.tweaker(arg))
            else:
                tweakers.append(q)

            for tweaker in tweakers:
                msg.debug('   [ %s ]'%tweaker.__name__)

                # treat arguments
                res = tweaker.arg.Treat()
                utl.CheckRes('\t- Treat Arguments', res)

                if res == var.SUCCESS:
                    utl.CheckRes('\t- DoIt', tweaker.DoIt())

                # utl.CheckRes('\t- Treat Arguments', tweaker.arg.Treat())
                #
                # utl.CheckRes('\t- DoIt', tweaker.DoIt())
