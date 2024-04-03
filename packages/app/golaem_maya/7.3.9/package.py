name = 'golaem_maya'
version = '7.3.9'

# variants = [
#     ['maya-2018']
# ]

requires = [
    'maya-2018'
]

def commands():
    import os

    INSTALL_PATH='/opt/plugins/golaem/Golaem-{0}-Maya2018'.format(version)
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Golaem {}.'.format(version))

    env.GOLAEM_VER.set(version)
    env.MAYA_MODULE_PATH.append(INSTALL_PATH)
    env.golaem_LICENSE.set('5053@10.10.10.129')

    env.PYTHONPATH.append('{base}/scripts')

    alias('golaem', 'maya')
