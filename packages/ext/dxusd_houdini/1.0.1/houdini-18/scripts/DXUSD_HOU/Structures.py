#coding:utf-8
from __future__ import print_function

import DXUSD.moduleloader as mdl
import DXUSD.Structures as srt
mdl.importModule(__name__, srt)

import DXUSD_HOU.Vars as var

if msg.DEV:
    reload(var)


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
        for item in self.items:
            res += str(item)
        return res

    @property
    def items(self):
        keys = self._items.keys()
        keys.sort()
        return [self._items[v] for v in keys]

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
            res = '[%s (%s)] - %s\n'%(self.arg.task, self.code, str(self.vers))
            res += '\t%s'%str(self.arg)
            res += Items.__repr__(self)
            return res


    class NSLayer(Items):
        def __init__(self, name, parent):
            Items.__init__(self, name, parent)
            self._child = Tasks.Layer
            self.vers = list()
            self.arg  = parent.arg

        def __repr__(self):
            res = '\tN : %s - %s\n'%(self.name, str(self.vers))
            res += Items.__repr__(self)
            return res


    class Layer:
        def __init__(self, name, parent):
            self.name = name
            self._parent = parent
            self.arg  = parent.arg

            self.lyrtask = None
            self.cliprate = None

            self.outpath   = ''
            self.inputnode = ''
            self.inputtype = ''
            self.dependency = {}

        def __repr__(self):
            if not self.outpath or not self.name:
                res = '\t\t >>> Cannot resolbe output file path\n'
            else:
                res  = '\t\t >>> %s\n'%utl.SJoin(self.outpath, self.name)
            res += '\t\t    Layer Type : %s\n'%str(self.inputtype)
            res += '\t\t    Layer Task : %s\n'%str(self.lyrtask)
            if self.cliprate:
                res += '\t\t    Clip Rate : %s\n'%self.cliprate
            if self.inputnode:
                res += '\t\t    Input Node : %s\n'%self.inputnode
            if self.dependency:
                for kind, data in self.dependency.items():
                    if data.has_key(var.USDPATH):
                        usdpath = data[var.USDPATH]
                        res += '\t\t    Depend %s USD : %s\n'%(kind, usdpath)
                    res += '\t\t    Depend %s Extra Variants :\n'%kind
                    if data[var.ORDER]:
                        for k in data[var.ORDER]:
                            res += '\t\t\t    %s : %s\n'%(k, data[k])
            return res
