name = 'Zelos'

requires = [
    # 'gcc-6.3.1',
    'glew',
#    'oiio-2.1.16',
    'oiio-1.8.9',
    'boost-1.61.0',
    'alembic',
    'extern'

]

variants = [
    ['maya-2018', 'usd_maya-19.06'],
    ['maya-2018', 'usd_maya-19.11'],
    ['maya-2022', 'usd_maya-21.00']
]

def commands():
    env.LD_LIBRARY_PATH.append('{}/lib'.format(env.MAYA_LOCATION))
    env.LD_LIBRARY_PATH.append('{}/lib2'.format(env.MAYA_LOCATION))
    env.LD_LIBRARY_PATH.append('{}/plug-ins/xgen/lib'.format(env.MAYA_LOCATION))
    env.LD_LIBRARY_PATH.append('{root}/lib')


    env.MAYA_MODULE_PATH.append('{root}')
    env.LD_LIBRARY_PATH.append('{root}/python')

