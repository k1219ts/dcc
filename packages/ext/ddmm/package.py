# -*- coding: utf-8 -*-

name = 'ddmm'

requires = [
    'pyside2',
    'ffmpeg_toolkit'
]

def commands():
    env.PATH.append('{this.root}/bin')
    env.PYTHONPATH.append('{this.root}/scripts')
