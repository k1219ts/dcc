name = 'Tane'

requires = [
#    'gcc-4.8.5',
#    'renderman-23.5',
    'tbb-2017_U6',
    'alembic',
    'eigen',
    'zelos'
]

variants = [
    ['maya-2018', 'usd_maya-19.05'],
    ['maya-2018', 'usd_maya-19.06'],
    ['maya-2018', 'usd_maya-19.11'],
    ['maya-2022', 'usd_maya-21.00']
]

def commands():
    # import os
    # glpath = '/netapp/backstage/pub/lib/Tane/1.0.0702/19.05/resource/baseGl'
    # if not os.path.exists(glpath):
    #    stop('Not installed Tane resource.')

    env.MAYA_MODULE_PATH.append('{root}')
