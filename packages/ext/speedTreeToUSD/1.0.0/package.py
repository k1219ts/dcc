name = 'speedTreeToUSD'
version = '1.0.0'

requires = [
    'pyside2',
    'pyalembic-1.7.1',
    'usd_core',
    'dxusd-2.0'
]

tools = [
    'speedTreeToUSD'
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
