name = 'dxpublisher'

requires = [
    'maya-2018',
    'usd_core',
    'dxusd',
    'dxusd_maya',
    'dxrulebook',
    'python-2',
    'pyside2-5.12.6',
    'pylibs-2'
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
