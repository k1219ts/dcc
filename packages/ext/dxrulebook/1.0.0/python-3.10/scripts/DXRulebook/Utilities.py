#coding:utf-8
'''
    Copyright & Author: 2016, Sehwi Park <sehwida@gmail.com>

    Description:
        useful utilities for the project
'''

from __future__ import print_function
import os
import re

import collections
import functools

def libpath(*subpath):
    '''
    Args:
        *subpath: list of subpath tokens
    '''
    library_path = re.sub('/python/.*', '/', os.path.abspath(__file__))

    return os.path.join(library_path, *subpath)


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):

        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections._collections_abc.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

    pass
