name = 'physx'

version = '3.4'

variants = [
    ['maya-2018'],
    ['maya-2022']
]

def commands():
    env.MAYA_MODULE_PATH.append('{root}')
    env.LD_LIBRARY_PATH.append('{root}/ContentCore')
    env.LD_LIBRARY_PATH.append('{root}/ContentCore/PhysX_3.X')
