name = 'golaem_katana'
version = '7.2.2'

def commands():
    import os
    INSTALL_PATH='/opt/plugins/golaem/Golaem-7.2.2-Maya2018'
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Golaem 7.2.2.')

    env.GOLAEM_VER.set(version)
    env.golaem_LICENSE.set('5053@10.10.10.129')
    env.LD_LIBRARY_PATH.append('{}/lib'.format(INSTALL_PATH))
    env.KATANA_RESOURCES.append('{}/procedurals/katana'.format(INSTALL_PATH))
    env.KATANA_POST_PYTHONPATH.append('{}/procedurals/katana/Python'.format(INSTALL_PATH))

    # for RenderMan
    if 'renderman' in resolve:
        env.RMAN_DSOPATH.append('{}/procedurals'.format(INSTALL_PATH))
        env.RMAN_RIXPLUGINPATH.append('{}/shaders'.format(INSTALL_PATH))
