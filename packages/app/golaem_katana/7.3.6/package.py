name = 'golaem_katana'
version = '7.3.6'

def commands():
    import os
    INSTALL_PATH='/opt/plugins/golaem/Golaem-7.3.6-Katana3.5-linux'
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Golaem 7.3.6.')

    env.GOLAEM_VER.set(version)
    env.golaem_LICENSE.set('5053@10.10.10.129')
    env.LD_LIBRARY_PATH.append('{}/lib'.format(INSTALL_PATH))
    env.KATANA_RESOURCES.append('{}/procedurals/katana'.format(INSTALL_PATH))
    env.KATANA_POST_PYTHONPATH.append('{}/procedurals/katana/Python'.format(INSTALL_PATH))

    # for RenderMan
    if 'renderman' in resolve:
        env.RMAN_DSOPATH.append('{}/procedurals/renderman'.format(INSTALL_PATH))
        env.RMAN_RIXPLUGINPATH.append('{}/shaders/renderman'.format(INSTALL_PATH))
