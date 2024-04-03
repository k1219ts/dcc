name = 'assetbrowser2'

requires = [
    'baselib-2',
    'pylibs-2.7'
]


def commands():
    env.PYTHONPATH.append('{root}/scripts/')
    env.PATH.append('{root}/bin')
