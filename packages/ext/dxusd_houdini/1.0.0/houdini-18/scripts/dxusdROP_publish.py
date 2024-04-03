#coding:utf-8
from __future__ import print_function
import os
import DXRulebook.Interface as rb
import DXUSD.Utils as utl
import hou

NULL = '__NULL__'

class Tasks(object):
    def __init__(self, arg):
        self._tasks = dict()
        self._idx   = 0
        self.arg    = arg

    def __getitem__(self, k):
        if k in self._tasks.keys():
            return self._tasks[k]
        elif isinstance(k, int):
            return self._tasks[self.tasks[k]]
        else:
            raise KeyError('Tasks has no given key (%s)'%str(k))

    def __len__(self):
        return len(self._tasks)

    def __next__(self):
        if self._idx < len(self):
            res = self._tasks[self.tasks[self._idx]]
            self._idx += 1
            return res
        else:
            self._idx = 0
            raise StopIteration

    def __repr__(self):
        res  = '###### Tasks Result ######\n'
        res += 'Args : %s\n\n'%str(self.arg)
        for task in self:
            res += '%s\n'%str(task)
        return res

    @property
    def tasks(self):
        keys = self._tasks.keys()
        keys.sort()
        return keys


    def Add(self, name, code):
        if name not in self._tasks.keys():
            self._tasks[name] = Tasks.Task(name, self, code)
        return self._tasks[name]


    class Task:
        def __init__(self, name, parent, code):
            self.code = code
            self.vers = list()

            self.arg  = rb.Flags('USD', **parent.arg)
            self.arg.task = name

            self._groups = dict()
            self._parent = parent


        def __getitem__(self, k):
            if k in self._groups.keys():
                return self._groups[k]
            else:
                return self.Add(k)

        def __repr__(self):
            res = '[%s (%s)] - %s\n'%(self.arg.task, self.code, str(self.vers))
            for group in self._groups:
                res += str(group)
            return res

        @property
        def groups(self):
            return self._groups.keys()


        def Add(self, name):
            if name not in self._groups.keys():
                self._groups[name] = Tasks.LayerGroup(name, self)


    class LayerGroup:
        def __init__(self, name, parent):
            self.name = name
            self.vers = list()
            self._layers = dict()
            self._parent = parent

        def __repr__(self):
            res = '\tGroup : %s - %s\n'%(self.name, str(self.vers))
            return res


    class Layer:
        def __init__(self):
            self.output = None
            self.inputnode = None

        def __repr__(self):
            return ''


class ROPLayers:
    def __init__(self):
        self._lyrs = list()
        self._idx  = 0

    def __len__(self):
        return len(self._lyrs)

    def __next__(self):
        if self._idx < len(self):
            res = self._lyrs[self._idx]
            self._idx += 1
            return res
        else:
            self._idx = 0
            raise StopIteration

    def __getitem__(self, k):
        return self._lyrs[k]

    def __repr__(self):
        res = ''
        for lyr in self._lyrs:
            res += '%s\n'%str(lyr)
        return res

    def Add(self, path):
        layer = ROPLayers.ROPLayer(path)
        self._lyrs.append(layer)
        return layer

    class ROPLayer:
        def __init__(self, path):
            self.path   = path
            self.type   = None
            self.frames = None    # bool
            self.clips  = None     # tuple (0.8, 1.0, 1.5)
            self.fxlyr  = None     # string
            self.lyrgrp = None

        def __repr__(self):
            return ('%s : %s (frame : %s, clips : %s, fxlyr : %s)'%(self.path,
                    self.type, str(self.frames),
                    str(self.clips), str(self.fxlyr)))

        def GetTaskCode(self, kind):
            '''
            asset : kind = 0
            shot  : kind = 1
            fx    : kind = 2
            '''
            res = (None, '')
            if self.type == 'geom':
                pass
            elif self.type == 'inst':
                res = [('model', 'V'), ('layout', 'NV'), ('fx', 'VNV')][kind]
            elif self.type == 'groom':
                pass
            elif self.type == 'crowd':
                pass

            return res[0], 'TASK%s'%res[1]


def ResolveTasks(arg, lyrs):
    '''
    [Arguments]
    arg  (DXRulebook.Interface.Flags())
    lyrs (ROPLayers)
    '''
    tasks = Tasks(arg)
    if arg.has_key('asset'):
        kind = 0
    elif arg.has_key('shot'):
        kind = 1
    else:
        return tasks

    for lyr in lyrs:
        taskname, code = lyr.GetTaskCode(kind if not lyr.fxlyr else 2)

        # ----------------------------------------------------------------------
        # Add or find task
        if taskname in tasks.tasks:
            task = tasks[taskname]
        else:
            task = tasks.Add(taskname, code)

            # check task version
            if code.startswith('TASKV'):
                vers = sorted(utl.GetVersions(task.arg.D.TASK))
                vers.reverse()
                nver = utl.Ver(utl.VerAsInt(vers[0])+1) if vers else utl.Ver(1)
                vers.insert(0, nver)
                task.vers = vers

        # ----------------------------------------------------------------------
        # Add or find layer group (nslyr)
        grp = task.Add(lyr.lyrgrp if 'N' in code else NULL)



    return tasks


if __name__ == '__main__':
    # lyrs = TaskLayers()
    # print lyrs['model'].name
    lyrs = ROPLayers()
    lyrs.Append('../asdf', 'set')
    lyrs.Append('../asdf', 'set')
    lyrs.Append('../asdf', 'set')

    for lyr in lyrs:
        print(lyr)
