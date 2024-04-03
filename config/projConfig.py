#!/bin/python

import os, sys
import argparse

_PROC_MAP_ = {
    'maya': 'maya',
    'mayapy': 'maya',
    'houdini': 'houdini',
    'hython': 'houdini',
    'katana': 'katana',
    'rfk': 'katana',
    'nuke': 'nuke',
    'nukeX': 'nuke',
    'nukeS': 'nuke',
    'nukeP': 'nuke',
}

def maya_config(args):
    cfg = ''
    if args.show:
        rootDir = '/show/{}/_config/maya'.format(args.show)
        cfg = '{}/scripts'.format(rootDir)
        if args.seq:
            cfg = '{ROOT}/{SEQ}/scripts'.format(ROOT=rootDir, SEQ=args.seq)
        if args.shot:
            cfg = '{ROOT}/{SEQ}/{SHOT}/scripts'.format(ROOT=rootDir, SEQ=args.shot.split('_')[0], SHOT=args.shot)
        cfg = 'PROJECTCONFIG=%s' % cfg
    return cfg

def houdini_config(args):
    cfg = ''
    if args.show:
        rootDir = '/show/{}/_config/houdini'.format(args.show)
        pkgPath = '{}/hfs'.format(rootDir)
        if args.seq:
            pkgPath = '{ROOT}/{SEQ}/hfs'.format(ROOT=rootDir, SEQ=args.seq)
        if args.shot:
            pkgPath = '{ROOT}/{SEQ}/{SHOT}/hfs'.format(ROOT=rootDir, SEQ=args.shot.split('_')[0], SHOT=args.shot)

        cfg = 'REZ_PACKAGES_PATH=' + pkgPath
        if os.getenv('REZ_PACKAGES_PATH'):
            cfg += ':' + os.getenv('REZ_PACKAGES_PATH')
    return cfg

def katana_config(args):
    cfg = ''
    if args.show:
        rootDir = '/show/{}/_config/katana'.format(args.show)
        cfg = '{}/Resources'.format(rootDir)
        if args.seq:
            cfg = '{ROOT}/{SEQ}/Resources'.format(ROOT=rootDir, SEQ=args.seq)
        if args.shot:
            cfg = '{ROOT}/{SEQ}/{SHOT}/Resources'.format(ROOT=rootDir, SEQ=args.shot.split('_')[0], SHOT=args.shot)
        cfg = 'PROJECTCONFIG=%s' % cfg
    return cfg

def nuke_config(args):
    cfg = ''
    if args.show:
        rootDir = '/show/{}/_config/nuke'.format(args.show)
        cfg = '{}/scripts'.format(rootDir)
        if args.seq:
            cfg = '{ROOT}/{SEQ}/scripts'.format(SHOW=args.show, SEQ=args.seq)
        if args.shot:
            cfg = '{ROOT}/{SEQ}/{SHOT}/scripts'.format(ROOT=rootDir, SEQ=args.shot.split('_')[0], SHOT=args.shot)
        cfg = 'PROJECTCONFIG=%s' % cfg
    return cfg


if __name__ == '__main__':
    opts = list(sys.argv)
    if '--help' in opts:
        opts.remove('--help')
    if '-h' in opts:
        opts.remove('-h')

    parser = argparse.ArgumentParser(description='Show Configuration')

    parser.add_argument('--show', type=str)
    parser.add_argument('--seq', type=str)
    parser.add_argument('--shot', type=str)

    args, unknown = parser.parse_known_args(opts)

    proc = None
    for n in unknown:
        if n in _PROC_MAP_.keys():
            proc = _PROC_MAP_[n]
            break
        if n.startswith('maya-'):
            proc = 'maya'
            break
        if n.startswith('houdini-'):
            proc = 'houdini'
            break
        if n.startswith('katana-'):
            proc = 'katana'
            break
        if n.startswith('nuke-'):
            proc = 'nuke'
            break

    if proc:
        proc = _PROC_MAP_[proc]
        cfg  = eval('%s_config(args)' % proc)
        sys.exit(cfg)
    else:
        sys.exit('')
