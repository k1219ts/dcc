name = 'otiotoolkit'
version = '1.0'

requires = [
    'otio',
    'oiio',
    'python-2.7',
    'pylibs-2.7',
    'pyside2-5.12.6',
    'baselib-2.5',
    # 'tesseract',
    'ffmpeg_toolkit',
    'dxrulebook'
]

def commands():
    env.PATH.append('{root}/bin')
    env.PYTHONPATH.append('{root}/scripts')
    env.OTIOTOOLKIT_SCRIPT_PATH.set('{root}/scripts')
