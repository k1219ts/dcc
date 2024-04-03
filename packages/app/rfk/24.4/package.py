name = 'rfk'
version = '24.4'

requires = [
    'renderman-24.4',
    'dxusd_katana'
]

variants = [
    ['katana-4.0'],
    ['katana-5.0']
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

    alias('rfk', 'katana')
