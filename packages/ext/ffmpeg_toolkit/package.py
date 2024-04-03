# -*- coding: utf-8 -*-

name = 'ffmpeg_toolkit'

requires = [
    'otio',
    'ffmpeg-4.2.0',
    'python-2'
]

def commands():
    env.PATH.append('{this.root}/bin')
    env.PYTHONPATH.append('{this.root}/scripts')
