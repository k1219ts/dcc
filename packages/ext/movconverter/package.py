# -*- coding: utf-8 -*-

name = 'movconverter'

requires = [
    'pyside2-5.12.6',
    'ffmpeg_toolkit'
]

def commands():
    env.PATH.append('{this.root}/bin')
    env.PYTHONPATH.append('{this.root}/scripts')
