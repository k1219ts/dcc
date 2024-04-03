name = 'maya_animation'

variants = [
    ['maya-2018'],
    ['maya-2022']
]

def commands():
    env.MAYA_MODULE_PATH.append('{root}')
    env.PYTHONPATH.append('{root}/scripts/studioLib')
