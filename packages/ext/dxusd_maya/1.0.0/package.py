name = 'dxusd_maya'
version = '1.0.0'

variants = [
    ['maya-2018'],
    ['maya-2022']
]

requires = [
    'dxusd',
    'python-2'
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
    env.MAYA_MODULE_PATH.append('{root}')
