name = 'rfk'
version = '25.2'

requires = [
    'renderman-25.2',
    'dxusd_katana'
]

variants = [
    ['katana-6.0']
]

def commands():
    import os

    env.RFKTREE.set('/opt/pixar/RenderManForKatana-{0}'.format(version))
    if not os.path.exists(str(env.RFKTREE)):
        stop('Not installed RenderManForKatana-{0}'.format(version))

    env.KATANA_RESOURCES.append('{}/plugins/katana{}'.format(env.RFKTREE, getenv("REZ_KATANA_VERSION")))

    env.DEFAULT_RENDERER.set('prman')

    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.LUA_PATH.append('{root}/lua/?.lua')

    # stdrepo path
    env.KATANA_RESOURCES.append('/stdrepo/LNR/97_DCC/app/rfk/25.2/katana-6.0/Resources')
    env.PYTHONPATH.append('/stdrepo/LNR/97_DCC/app/rfk/25.2/katana-6.0/scripts')

    alias('rfk', 'katana')
