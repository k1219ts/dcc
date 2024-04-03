name = 'maya_layout'

variants = [
    ['maya-2018'],
    ['maya-2022']
]

def commands():
    env.MAYA_MODULE_PATH.append('{root}')
