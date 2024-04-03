name = 'beyondscreen'
version = '1.0.0'

requires = [
    'openexr-2.2'
]

variants = [
    ['maya-2018']
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.MAYA_MODULE_PATH.append('{root}')
