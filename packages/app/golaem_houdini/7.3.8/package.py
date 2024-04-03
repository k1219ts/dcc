name = 'golaem_houdini'
version = '7.3.8'

def commands():
    import os
    INSTALL_PATH='/opt/plugins/golaem/Golaem-7.3.8-Houdini18.5-linux'
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Golaem 7.3.8.')

    env.GOLAEM_VER.set(version)
    env.golaem_LICENSE.set('5053@10.10.10.129')
    env.LD_LIBRARY_PATH.append('{}/lib'.format(INSTALL_PATH))
    env.HOUDINI_DSO_PATH.append('{}/procedurals/houdini'.format(INSTALL_PATH))
    env.PYTHONPATH.append('{}/procedurals/houdini/python'.format(INSTALL_PATH))

