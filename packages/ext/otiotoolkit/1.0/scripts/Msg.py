#coding:utf-8
from __future__ import print_function
from Define import Colors
import types


def warning(*args, **kwargs):
    args = list(args)
    args.insert(0, '# Warning :')
    args.insert(0, Colors.WARNING)
    args.append(Colors.ENDC)

    print(*args, **kwargs)


def errorMsg(*args, **kwargs):
    args = list(args)
    args.insert(0, '# Error :')
    args.insert(0, Colors.ERROR)
    args.append(Colors.ENDC)

    print(*args, **kwargs)

def error(*args, **kwargs):
    args = list(args)
    args.insert(0, '# Error :')
    args.insert(0, Colors.ERROR)
    args.append(Colors.ENDC)

    print(*args, **kwargs)


def bold(*args, **kwargs):
    args = list(args)
    args.insert(0, Colors.BOLD)
    args.append(Colors.ENDC)

    print(*args, **kwargs)