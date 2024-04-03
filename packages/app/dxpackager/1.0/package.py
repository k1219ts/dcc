name = 'dxpackager'
version = '1.0'

requires = [
    'usd_core-20.08',
    'dxusd',
    'dxrulebook',
    'python-2',
    'pyside2-5.12',
    'pylibs-2'
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
