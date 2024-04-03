name = 'dxusd_houdini'

requires = [
    'dxusd',
    'dxrulebook'
]

variants = [
    ['houdini-18']
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.HOUDINI_PATH.append('{root}')
    env.PATH.append("{root}/bin")