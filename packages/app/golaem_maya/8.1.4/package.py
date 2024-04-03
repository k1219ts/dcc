name = 'golaem_maya'
version = '8.1.4'

# variants = [
#     ['maya-2018']
# ]

requires = [
    'physx'
]

def commands():
    import os

    INSTALL_PATH='/opt/plugins/golaem/Golaem-{0}-Maya{1}'.format(version, getenv('MAYA_VER'))
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Golaem {}.'.format(version))

    env.GOLAEM_VER.set(version)
    env.MAYA_MODULE_PATH.append(INSTALL_PATH)
    env.golaem_LICENSE.set('5053@10.10.10.129')

    env.PYTHONPATH.append('{base}/scripts')

    alias('golaem', 'maya')
