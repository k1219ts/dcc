#coding:utf-8
from __future__ import print_function

import DXUSD.moduleloader as md
import DXUSD.Structures as srt
md.importModule(__name__, srt)

import DXUSD.Message as msg
import DXUSD_HOU.Vars as var


class Items(object):
    def __init__(self, name=None, parent=None):
        self.name = name
        self._items = dict()
        self._idx = 0
        self._parent = parent # class instance
        self._child  = None   # class type

    def __getitems__(self, k):
        if k in self._items.keys():
            return self._items[k]
        elif isinstance(k, int):
            return self.items[k]
        else:
            return self.self._child(k)

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        res = ''
        for item in self.items():
            res += str(item)
        return res

    def keys(self):
        keys = self._items.keys()
        keys.sort()
        return keys

    def items(self):
        return [self._items[v] for v in self.keys()]

    def parent(self):
        return self._parent

    def Add(self, name):
        if name not in self._items.keys():
            self._items[name] = self._child(name, self)
        return self._items[name]


class Tasks(Items):
    def __init__(self, arg):
        Items.__init__(self)
        self._child = Tasks.Task
        self.arg    = arg

    def __repr__(self):
        res  = '###### Tasks Result ######\n'
        # res += 'Args : %s\n\n'%str(self.arg)
        res += Items.__repr__(self)
        return res

    def Add(self, name, code):
        item = Items.Add(self, name)
        item.code = code
        return item

    class Task(Items):
        def __init__(self, name, parent):
            Items.__init__(self, name, parent)
            self._child = Tasks.NSLayer
            self.code = None
            self.vers = list()
            self.arg  = Arguments(**parent.arg)
            self.arg.task = name

        def __repr__(self):
            res = '[ %s (%s) ] - %s\n'
            res = res%(self.arg.task, self.code, str(self.vers))
            res += '\t%s'%str(self.arg)
            res += Items.__repr__(self)
            return res


    class NSLayer(Items):
        def __init__(self, name, parent):
            Items.__init__(self, name, parent)
            self._child = Tasks.SubLayer
            self.vers = list()
            self.arg  = parent.arg

        def __repr__(self):
            res = ''
            if self.name != var.NULL:
                res += '\n\t[ NS Layer ]\n'
                res += '\t %s - %s\n'%(self.name, str(self.vers))
            res += Items.__repr__(self)
            return res


    class SubLayer(Items):
        def __init__(self, name, parent):
            Items.__init__(self, name, parent)
            self._child = Tasks.Layer
            self.vers = list()
            self.arg  = parent.arg

        def __repr__(self):
            res = ''
            if self.name != var.NULL:
                res += '\n\t[ Sub Layer ]\n'
                res += '\t %s - %s\n'%(self.name, str(self.vers))
            res += Items.__repr__(self)
            return res


    class Layer:
        def __init__(self, name, parent):
            self.name = name
            self._parent = parent
            self.arg  = parent.arg

            self.cliprate = None

            self.outpath   = ''
            self.inputnode = ''
            self.lyrtype = ''
            self.prctype = ''
            self.dependency = {}

        @property
        def task(self):
            return self._parent._parent._parent

        def __repr__(self):
            res = '\n\t[ Layers ]\n'
            if not self.outpath or not self.name:
                res += '\t >>> Cannot resolbe output file path\n'
            else:
                res += '\t >>> %s\n'%utl.SJoin(self.outpath, self.name)

            res += '\t    Layer Type : %s\n'%str(self.lyrtype)
            res += '\t    Post Process : %s\n'%str(self.prctype)
            if self.prctype == var.PRCCLIP:
                res += '\t    Clip Rate : %s\n'%self.cliprate

            if self.inputnode:
                res += '\t    Input Node : %s\n'%self.inputnode

            if self.dependency:
                res += '\n\t[Dependency]\n'
                for kind, tasks in self.dependency.items():
                    for task, data in tasks.items():
                        res += '\t    > %s:%s \n'%(kind, task)
                        if data.has_key(var.USDPATH):
                            usdpath = data[var.USDPATH]
                            res += '\t\tUSD Path : %s\n'%usdpath
                        res += '\t\tVariants :\n'
                        if data[var.ORDER]:
                            for k in data[var.ORDER]:
                                res += '\t\t  - %s : %s\n'%(k, data[k])
            return res
